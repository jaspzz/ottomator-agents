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

# New models for context API
class ContextRequest(BaseModel):
    query: str
    context_type: Optional[str] = "auto"  # auto, vector, graph, hybrid
    limit: Optional[int] = 5
    session_id: Optional[str] = None

class ContextItem(BaseModel):
    content: str
    source: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class ContextResponse(BaseModel):
    results: List[ContextItem]
    query: str
    total_results: int
    context_type: str
    metadata: Optional[Dict[str, Any]] = None

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

@app.post("/search/vector")
async def vector_search(
    request: ContextRequest,
    authenticated: bool = Depends(verify_auth)
):
    """Vector-only search for semantic similarity"""
    try:
        # You can implement direct vector search here
        # For now, use the agent with a vector-focused prompt
        result = await rag_agent.run(
            f"Search for documents semantically similar to: {request.query}",
            deps=AgentDependencies(session_id=request.session_id) if AgentDependencies else None
        )
        
        # Process and return results similar to get_context
        context_items = []
        sources = getattr(result, 'sources', [])
        
        for source in sources[:request.limit]:
            context_items.append(ContextItem(
                content=source.get('content', ''),
                source=source.get('source', 'Vector Search'),
                score=source.get('similarity'),
                metadata={'search_type': 'vector', **source.get('metadata', {})}
            ))
        
        return ContextResponse(
            results=context_items,
            query=request.query,
            total_results=len(context_items),
            context_type="vector"
        )
        
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/graph")
async def graph_search(
    request: ContextRequest,
    authenticated: bool = Depends(verify_auth)
):
    """Knowledge graph search for relationships"""
    try:
        # Graph-focused search
        result = await rag_agent.run(
            f"Find relationships and connections related to: {request.query}",
            deps=AgentDependencies(session_id=request.session_id) if AgentDependencies else None
        )
        
        context_items = []
        sources = getattr(result, 'sources', [])
        
        for source in sources[:request.limit]:
            context_items.append(ContextItem(
                content=source.get('content', ''),
                source=source.get('source', 'Knowledge Graph'),
                score=source.get('confidence'),
                metadata={'search_type': 'graph', **source.get('metadata', {})}
            ))
        
        return ContextResponse(
            results=context_items,
            query=request.query,
            total_results=len(context_items),
            context_type="graph"
        )
        
    except Exception as e:
        logger.error(f"Graph search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context/summary")
async def get_context_summary(
    request: ContextRequest,
    authenticated: bool = Depends(verify_auth)
):
    """Get a summarized context for the query"""
    try:
        # Get detailed context first
        context_response = await get_context(request, authenticated)
        
        # Create a summary from the results
        if not context_response.results:
            return {
                "summary": f"No relevant information found for: {request.query}",
                "query": request.query,
                "sources_count": 0
            }
        
        # Combine all content for summary
        combined_content = "\n\n".join([
            f"Source: {item.source}\nContent: {item.content}"
            for item in context_response.results
        ])
        
        return {
            "summary": combined_content[:2000],  # Limit length
            "query": request.query,
            "sources_count": len(context_response.results),
            "sources": [item.source for item in context_response.results]
        }
        
    except Exception as e:
        logger.error(f"Summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/context", response_model=ContextResponse)
async def get_context(
    request: ContextRequest,
    authenticated: bool = Depends(verify_auth)
):
    """Get dynamic context for TypingMind agents"""
    try:
        if not rag_agent:
            raise HTTPException(status_code=500, detail="RAG agent not available")
        
        # Use the RAG agent's tools to get context
        if AgentDependencies:
            deps = AgentDependencies(session_id=request.session_id)
            
            # Run the agent but extract the sources/context instead of the full response
            result = await rag_agent.run(
                f"Find relevant information about: {request.query}", 
                deps=deps
            )
        else:
            result = await rag_agent.run(f"Find relevant information about: {request.query}")
        
        # Extract context items from the result
        context_items = []
        sources = getattr(result, 'sources', [])
        
        # Convert agent sources to context items
        for i, source in enumerate(sources[:request.limit]):
            context_items.append(ContextItem(
                content=source.get('content', ''),
                source=source.get('source', f'Document {i+1}'),
                score=source.get('similarity', None),
                metadata={
                    'document_title': source.get('document_title'),
                    'chunk_id': source.get('chunk_id'),
                    'type': source.get('type', 'unknown')
                }
            ))
        
        return ContextResponse(
            results=context_items,
            query=request.query,
            total_results=len(context_items),
            context_type=request.context_type or "auto",
            metadata={
                "tools_used": getattr(result, 'tools_used', []),
                "processing_time": getattr(result, 'processing_time', None)
            }
        )
        
    except Exception as e:
        logger.error(f"Context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=ContextResponse)  
async def search_knowledge(
    request: ContextRequest,
    authenticated: bool = Depends(verify_auth)
):
    """Search knowledge base - alias for context endpoint"""
    return await get_context(request, authenticated)

@app.get("/context/test")
async def test_context(
    query: str = "Microsoft AI initiatives",
    authenticated: bool = Depends(verify_auth)
):
    """Test the context API with a simple query"""
    request = ContextRequest(query=query, limit=3)
    return await get_context(request, authenticated)

@app.get("/health")
async def health():
    """Health check with context API status"""
    return {
        "status": "CONTEXT_API_READY",
        "service": "agentic-rag-context-api",
        "version": "1.0.0-context",
        "auth_enabled": os.getenv('ENABLE_AUTH', 'false').lower() == 'true',
        "api_key_configured": bool(os.getenv('API_KEY')),
        "endpoints": [
            "/context",
            "/search", 
            "/search/vector",
            "/search/graph",
            "/context/summary",
            "/context/test"
        ],
        "integration": "typingmind-dynamic-context"
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
