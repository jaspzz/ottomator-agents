import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

# ============= INLINE AUTHENTICATION =============
security = HTTPBearer(auto_error=False)

async def verify_auth(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Inline authentication"""
    
    # Check if auth is enabled
    auth_enabled = os.getenv('ENABLE_AUTH', 'false').lower() == 'true'
    if not auth_enabled:
        return True
    
    # Get expected API key
    expected_key = os.getenv('API_KEY')
    if not expected_key:
        return True  # No key configured, allow access
    
    # Check Bearer token from security dependency
    if credentials and credentials.credentials == expected_key:
        return True
    
    # Check Authorization header manually
    if authorization and authorization.startswith('Bearer '):
        token = authorization[7:]  # Remove 'Bearer ' prefix
        if token == expected_key:
            return True
    
    # Check X-API-Key header
    if x_api_key and x_api_key == expected_key:
        return True
    
    # No valid auth found
    raise HTTPException(
        status_code=401,
        detail={
            "error": "Authentication required",
            "message": "Use Bearer token in Authorization header or X-API-Key header"
        }
    )

# ============= ORIGINAL IMPORTS (with error handling) =============
try:
    from .agent import rag_agent, AgentDependencies
except ImportError as e:
    logger.error(f"Could not import agent: {e}")
    rag_agent = None
    AgentDependencies = None

try:
    from .db_utils import DatabasePool
except ImportError as e:
    logger.error(f"Could not import db_utils: {e}")
    DatabasePool = None

try:
    from .graph_utils import initialize_graph
except ImportError as e:
    logger.error(f"Could not import graph_utils: {e}")
    async def initialize_graph():
        pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

class StreamChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# ============= ORIGINAL LIFESPAN LOGIC =============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events - using original pattern"""
    logger.info("Starting up agentic RAG API...")
    
    try:
        # Database initialization (from original code)
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
        # Simple cleanup - don't call functions that don't exist
        logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Agentic RAG API - Secured",
    description="RAG system with authentication",
    version="1.0.0-auth",
    lifespan=lifespan
)

# CORS Configuration for TypingMind
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.typingmind.com",
        "https://app.typingmind.com", 
        "https://typingmind.com",
        "http://localhost:3000",
        "http://localhost:8080",
        "https://z0s8gsogo48w4sg4o0sckss8.spielvogel.io",
        "*"  # For testing - remove in production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ============= ENDPOINTS =============

@app.get("/health")
async def health():
    """Health check with authentication status"""
    return {
        "status": "AUTH_VERSION_DEPLOYED",
        "service": "agentic-rag-secured", 
        "version": "1.0.0-auth",
        "auth_enabled": os.getenv('ENABLE_AUTH', 'false').lower() == 'true',
        "api_key_configured": bool(os.getenv('API_KEY')),
        "environment_debug": {
            "enable_auth": os.getenv('ENABLE_AUTH'),
            "api_key_length": len(os.getenv('API_KEY', ''))
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authenticated: bool = Depends(verify_auth)
):
    """Main chat endpoint with authentication"""
    try:
        if not rag_agent:
            raise HTTPException(status_code=500, detail="RAG agent not available")
        
        # Use original agent pattern if available
        if AgentDependencies:
            deps = AgentDependencies(
                session_id=request.session_id or request.conversation_id
            )
            result = await rag_agent.run(request.message, deps=deps)
        else:
            # Fallback if dependencies not available
            result = await rag_agent.run(request.message)
        
        return ChatResponse(
            response=result.data,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            sources=getattr(result, 'sources', []),
            metadata={
                "authenticated": True,
                "tools_used": getattr(result, 'tools_used', [])
            }
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(
    request: StreamChatRequest,
    authenticated: bool = Depends(verify_auth)
):
    """Streaming chat with authentication"""
    try:
        if not rag_agent:
            raise HTTPException(status_code=500, detail="RAG agent not available")
        
        async def generate_stream():
            try:
                if AgentDependencies:
                    deps = AgentDependencies(session_id=request.session_id)
                    async with rag_agent.run_stream(request.message, deps=deps) as result:
                        async for message in result.stream():
                            yield f"data: {message}\n\n"
                else:
                    # Fallback for simpler stream
                    async for message in rag_agent.run_stream(request.message):
                        yield f"data: {message}\n\n"
                        
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Simple test endpoint
@app.get("/test-auth")
async def test_auth(authenticated: bool = Depends(verify_auth)):
    return {
        "message": "Authentication is working!",
        "authenticated": True,
        "timestamp": "now"
    }

# Debug endpoint  
@app.get("/debug/env")
async def debug_env():
    return {
        "enable_auth": os.getenv('ENABLE_AUTH'),
        "api_key_set": bool(os.getenv('API_KEY')),
        "api_key_length": len(os.getenv('API_KEY', ''))
    }

if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", 8058))
    host = os.getenv("APP_HOST", "0.0.0.0")
    
    uvicorn.run(
        "agent.api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
