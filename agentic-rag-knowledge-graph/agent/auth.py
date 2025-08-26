import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# Import the auth middleware
from .auth import verify_api_key_multiple_methods, verify_api_key

# Import existing modules (adjust these imports based on the actual structure)
from .agent import rag_agent, AgentDependencies
from .db_utils import close_db_pool
from .graph_utils import initialize_graph, close_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000

class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

class StreamChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting up agentic RAG API...")
    
    try:
        # Initialize databases
        logger.info("Database initialized")
        
        # Initialize graph
        await initialize_graph()
        logger.info("Graph initialized")
        
        logger.info("✅ Agentic RAG API startup complete!")
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down...")
        await close_db_pool()
        await close_graph()

# Create FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="RAG system with knowledge graph capabilities - Secured",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration for TypingMind and Traefik
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # TypingMind domains
        "https://www.typingmind.com",
        "https://app.typingmind.com", 
        "https://typingmind.com",
        # Local development
        "http://localhost:3000",
        "http://localhost:8080",
        # Your domain (add your actual domain)
        "https://z0s8gsogo48w4sg4o0sckss8.spielvogel.io",
        # Wildcard for subdomains if needed
        "https://*.spielvogel.io"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type", 
        "X-API-Key",
        "X-Requested-With",
        "Accept",
        "Origin"
    ],
)

# Public health check (no auth required)
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "agentic-rag-api",
        "auth_enabled": os.getenv('ENABLE_AUTH', 'false').lower() == 'true'
    }

# Secured chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authenticated: bool = Depends(verify_api_key_multiple_methods)
):
    """Main chat endpoint with authentication"""
    try:
        # Create agent dependencies
        deps = AgentDependencies(
            session_id=request.session_id or request.conversation_id
        )
        
        # Run the RAG agent
        result = await rag_agent.run(request.message, deps=deps)
        
        return ChatResponse(
            response=result.data,
            conversation_id=request.conversation_id,
            session_id=request.session_id,
            sources=getattr(result, 'sources', []),
            metadata={
                "tools_used": getattr(result, 'tools_used', []),
                "model_used": getattr(result, 'model', 'unknown')
            }
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Secured streaming chat endpoint  
@app.post("/chat/stream")
async def chat_stream(
    request: StreamChatRequest,
    authenticated: bool = Depends(verify_api_key_multiple_methods)
):
    """Streaming chat endpoint with authentication"""
    try:
        async def generate_stream():
            deps = AgentDependencies(session_id=request.session_id)
            
            async with rag_agent.run_stream(request.message, deps=deps) as result:
                async for message in result.stream():
                    yield f"data: {message}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
        
    except Exception as e:
        logger.error(f"Stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alternative endpoint with Bearer token only (for strict clients)
@app.post("/chat/secure", response_model=ChatResponse)
async def chat_secure(
    request: ChatRequest,
    authenticated: bool = Depends(verify_api_key)  # Bearer token only
):
    """Chat endpoint with strict Bearer token authentication"""
    return await chat(request, authenticated)

if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", 8058))
    host = os.getenv("APP_HOST", "0.0.0.0")
    
    uvicorn.run(
        "agent.api:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )