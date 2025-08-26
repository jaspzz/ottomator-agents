import asyncio
import logging
import os
from fastapi import UploadFile, File, Form
from fastapi.responses import HTMLResponse
import aiofiles
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, HTTPException, Depends, Request, Header, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import aiofiles
from pathlib import Path

# Configure logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import auto-ingestion service with proper error handling
auto_ingestion_service = None
try:
    from ingestion.file_watcher import AutoIngestionService
    logger.info("✅ Auto-ingestion service available")
except ImportError as e:
    logger.warning(f"⚠️ Auto-ingestion service not available: {e}")
    AutoIngestionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global auto_ingestion_service
    
    logger.info("Starting up agentic RAG API...")
    
    try:
        logger.info("Database initialized")
        await initialize_graph()
        logger.info("Graph initialized")
        
        # Start auto-ingestion service with better error handling
        if AutoIngestionService:
            logger.info("🔧 Initializing auto-ingestion service...")
            try:
                auto_ingestion_service = AutoIngestionService()
                logger.info("🔧 Auto-ingestion service instance created")
                
                # Start it as a background task with proper error handling
                async def start_ingestion_service():
                    try:
                        logger.info("🚀 Starting auto-ingestion service background task...")
                        await auto_ingestion_service.start()
                    except Exception as e:
                        logger.error(f"❌ Auto-ingestion service failed: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # Create and start the task
                ingestion_task = asyncio.create_task(start_ingestion_service())
                logger.info("🤖 Auto-ingestion background task created")
                
                # Give it a moment to start
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize auto-ingestion service: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning("⚠️ AutoIngestionService not available")
        
        logger.info("✅ Agentic RAG API startup complete!")
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("Shutting down...")
        if auto_ingestion_service and hasattr(auto_ingestion_service, 'observer'):
            try:
                auto_ingestion_service.observer.stop()
                auto_ingestion_service.observer.join()
                logger.info("🛑 File watcher stopped")
            except Exception as e:
                logger.error(f"Error stopping file watcher: {e}")



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

# Add topic models
class TopicContextRequest(BaseModel):
    query: str
    topic: Optional[str] = None  # Topic slug
    context_type: Optional[str] = "auto"
    limit: Optional[int] = 5
    session_id: Optional[str] = None

class TopicInfo(BaseModel):
    name: str
    slug: str
    description: str
    document_count: int
    chunk_count: int

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
    request: TopicContextRequest,  # Updated to use TopicContextRequest
    authenticated: bool = Depends(verify_auth)
):
    """Get dynamic context with optional topic filtering"""
    try:
        # Build the search prompt with topic context
        search_prompt = f"Find relevant information about: {request.query}"
        if request.topic:
            search_prompt += f" (Focus on {request.topic} context)"
        
        # Use your RAG agent with topic-aware search
        if AgentDependencies:
            deps = AgentDependencies(
                session_id=request.session_id,
                context={'topic': request.topic}  # Pass topic to agent
            )
            result = await rag_agent.run(search_prompt, deps=deps)
        else:
            result = await rag_agent.run(search_prompt)
        
        # Process results (your existing logic)
        context_items = []
        sources = getattr(result, 'sources', [])
        
        for source in sources[:request.limit]:
            context_items.append(ContextItem(
                content=source.get('content', ''),
                source=source.get('source', 'Unknown'),
                score=source.get('similarity'),
                metadata={
                    'topic': request.topic,
                    'document_title': source.get('document_title'),
                    'chunk_id': source.get('chunk_id'),
                    **source.get('metadata', {})
                }
            ))
        
        return ContextResponse(
            results=context_items,
            query=request.query,
            total_results=len(context_items),
            context_type=request.context_type or "auto",
            metadata={
                "topic_filter": request.topic,
                "tools_used": getattr(result, 'tools_used', [])
            }
        )
        
    except Exception as e:
        logger.error(f"Context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topics", response_model=List[TopicInfo])
async def list_topics(authenticated: bool = Depends(verify_auth)):
    """List all available topics"""
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        topics = await conn.fetch("""
            SELECT 
                t.name, t.slug, t.description,
                COUNT(DISTINCT d.id) as document_count,
                COUNT(c.id) as chunk_count
            FROM topics t
            LEFT JOIN documents d ON t.id = d.topic_id
            LEFT JOIN chunks c ON t.id = c.topic_id
            GROUP BY t.id, t.name, t.slug, t.description
            ORDER BY t.name
        """)
        await conn.close()
        
        return [TopicInfo(**dict(topic)) for topic in topics]
        
    except Exception as e:
        logger.error(f"Topics list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/topics")
async def create_topic(
    topic: dict,
    authenticated: bool = Depends(verify_auth)
):
    """Create a new topic"""
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        result = await conn.fetchrow("""
            INSERT INTO topics (name, description, slug)
            VALUES ($1, $2, $3)
            ON CONFLICT (slug) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description
            RETURNING id, name, slug, description
        """, topic['name'], topic.get('description', ''), topic['slug'])
        await conn.close()
        
        return dict(result)
        
    except Exception as e:
        logger.error(f"Create topic error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context/{topic_slug}")
async def get_topic_context(
    topic_slug: str,
    request: ContextRequest,  # Use simpler request model
    authenticated: bool = Depends(verify_auth)
):
    """Get context for a specific topic"""
    topic_request = TopicContextRequest(
        query=request.query,
        topic=topic_slug,
        context_type=request.context_type,
        limit=request.limit,
        session_id=request.session_id
    )
    return await get_context(topic_request, authenticated)

@app.post("/context/auto-route")
async def auto_route_context(
    request: ContextRequest,
    authenticated: bool = Depends(verify_auth)
):
    """Automatically detect topic and route query"""
    try:
        # Simple keyword-based topic detection
        query_lower = request.query.lower()
        
        topic = None
        if any(term in query_lower for term in ['business central', 'dynamics', 'erp', 'finance', 'inventory']):
            topic = 'business-central'
        elif any(term in query_lower for term in ['ai', 'artificial intelligence', 'machine learning', 'openai', 'gpt']):
            topic = 'ai-research'
        elif any(term in query_lower for term in ['cloud', 'azure', 'aws', 'infrastructure']):
            topic = 'cloud-computing'
        
        # Create topic-aware request
        topic_request = TopicContextRequest(
            query=request.query,
            topic=topic,
            context_type=request.context_type,
            limit=request.limit,
            session_id=request.session_id
        )
        
        result = await get_context(topic_request, authenticated)
        
        # Add routing info to metadata
        result.metadata = result.metadata or {}
        result.metadata['detected_topic'] = topic
        result.metadata['routing_method'] = 'keyword-based'
        
        return result
        
    except Exception as e:
        logger.error(f"Auto-route error: {e}")
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
