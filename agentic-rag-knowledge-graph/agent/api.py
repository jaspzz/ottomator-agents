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
    """Inline authentication - no external imports needed"""
    
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
            "message": "Use Bearer token in Authorization header or X-API-Key header",
            "auth_enabled": auth_enabled,
            "expected_key_length": len(expected_key) if expected_key else 0
        }
    )

# ============= END AUTHENTICATION =============

# Import existing modules - keeping original imports
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting up agentic RAG API...")
    
    try:
        logger.info("Database initialized")
        await initialize_graph()
        logger.info("Graph initialized")
        logger.info("✅ Agentic RAG API startup complete!")
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        logger.info("Shutting down...")
        await close_db_pool()
        await close_graph()

# Create FastAPI app
app = FastAPI(
    title="Agentic RAG API - Secured",
    description="RAG system with authentication",
    version="1.0.0-auth",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.typingmind.com",
        "https://app.typingmind.com", 
        "https://typingmind.com",
        "http://localhost:3000",
        "http://localhost:8080",
        "https://z0s8gsogo48w4sg4o0sckss8.spielvogel.io"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ============= ENDPOINTS =============

@app.get("/health")
async def health():
    """Health check - shows auth status"""
    return {
        "status": "AUTH_BUILD_DEPLOYED",
        "service": "agentic-rag-secured",
        "version": "1.0.0-auth",
        "auth_enabled": os.getenv('ENABLE_AUTH', 'false').lower() == 'true',
        "api_key_configured": bool(os.getenv('API_KEY')),
        "enable_auth_value": os.getenv('ENABLE_AUTH'),
        "api_key_length": len(os.getenv('API_KEY', ''))
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authenticated: bool = Depends(verify_auth)
):
    """Secured chat endpoint"""
    try:
        deps = AgentDependencies(
            session_id=request.session_id or request.conversation_id
        )
        
        result = await rag_agent.run(request.message, deps=deps)
        
        return ChatResponse(
            response=result.data,
            conversation_id=request.conversation_id,
            session_id=request.session_id,
            sources=getattr(result, 'sources', []),
            metadata={
                "tools_used": getattr(result, 'tools_used', []),
                "authenticated": True
            }
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    authenticated: bool = Depends(verify_auth)
):
    """Secured streaming chat"""
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
            }
        )
        
    except Exception as e:
        logger.error(f"Stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Debug endpoint to test auth
@app.get("/test-auth")
async def test_auth(authenticated: bool = Depends(verify_auth)):
    return {"message": "Authentication working!", "authenticated": True}

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
