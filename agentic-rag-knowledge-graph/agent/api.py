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
    """Application lifespan events with detailed logging"""
    global auto_ingestion_service
    
    # This should appear in logs
    logger.info("🚀 LIFESPAN: Starting up agentic RAG API...")
    print("🚀 LIFESPAN: Starting up agentic RAG API...")  # Force print to ensure visibility
    
    try:
        # Database initialization
        logger.info("📊 LIFESPAN: Database initialized")
        print("📊 LIFESPAN: Database initialized")
        
        # Graph initialization  
        await initialize_graph()
        logger.info("🕸️ LIFESPAN: Graph initialized")
        print("🕸️ LIFESPAN: Graph initialized")
        
        # Auto-ingestion service initialization
        logger.info("🔧 LIFESPAN: Checking auto-ingestion service availability...")
        print("🔧 LIFESPAN: Checking auto-ingestion service availability...")
        
        logger.info(f"🔧 LIFESPAN: AutoIngestionService available: {AutoIngestionService is not None}")
        print(f"🔧 LIFESPAN: AutoIngestionService available: {AutoIngestionService is not None}")
        
        if AutoIngestionService:
            logger.info("🔧 LIFESPAN: Creating auto-ingestion service...")
            print("🔧 LIFESPAN: Creating auto-ingestion service...")
            
            try:
                auto_ingestion_service = AutoIngestionService()
                logger.info("✅ LIFESPAN: Auto-ingestion service instance created")
                print("✅ LIFESPAN: Auto-ingestion service instance created")
                
                # Start background task
                async def start_ingestion():
                    try:
                        logger.info("🚀 LIFESPAN: Starting auto-ingestion background task...")
                        print("🚀 LIFESPAN: Starting auto-ingestion background task...")
                        await auto_ingestion_service.start()
                    except Exception as e:
                        logger.error(f"❌ LIFESPAN: Auto-ingestion start failed: {e}")
                        print(f"❌ LIFESPAN: Auto-ingestion start failed: {e}")
                
                # Create background task
                task = asyncio.create_task(start_ingestion())
                logger.info("📋 LIFESPAN: Background task created")
                print("📋 LIFESPAN: Background task created")
                
            except Exception as e:
                logger.error(f"❌ LIFESPAN: Failed to create auto-ingestion service: {e}")
                print(f"❌ LIFESPAN: Failed to create auto-ingestion service: {e}")
                import traceback
                logger.error(f"🔍 LIFESPAN: Traceback: {traceback.format_exc()}")
                print(f"🔍 LIFESPAN: Traceback: {traceback.format_exc()}")
        else:
            logger.warning("⚠️ LIFESPAN: AutoIngestionService class not available")
            print("⚠️ LIFESPAN: AutoIngestionService class not available")
        
        logger.info("✅ LIFESPAN: Startup complete, yielding control...")
        print("✅ LIFESPAN: Startup complete, yielding control...")
        
        # Application runs here
        yield
        
    except Exception as e:
        logger.error(f"💥 LIFESPAN: Startup failed: {e}")
        print(f"💥 LIFESPAN: Startup failed: {e}")
        import traceback
        logger.error(f"🔍 LIFESPAN: Error traceback: {traceback.format_exc()}")
        print(f"🔍 LIFESPAN: Error traceback: {traceback.format_exc()}")
        raise
    finally:
        logger.info("🛑 LIFESPAN: Shutdown started")
        print("🛑 LIFESPAN: Shutdown started")

# Make sure the app uses this lifespan
app = FastAPI(
    title="Agentic RAG API - Secured",
    description="RAG system with authentication",
    version="1.0.0-auth",
    lifespan=lifespan  # Make sure this is set!
)



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

@app.get("/debug/check-directories")
async def check_directories(authenticated: bool = Depends(verify_auth)):
    """Check what files are actually in the watch directories"""
    watch_dir = Path("/app/ingestion-watch")
    
    result = {
        "watch_dir_exists": watch_dir.exists(),
        "watch_dir_path": str(watch_dir),
        "directories": {}
    }
    
    topics = ['business-central', 'ai-research', 'cloud-computing', 'general', 'processed', 'failed']
    
    for topic in topics:
        topic_dir = watch_dir / topic
        if topic_dir.exists():
            files = []
            try:
                for item in topic_dir.iterdir():
                    if item.is_file():
                        files.append({
                            "name": item.name,
                            "size": item.stat().st_size,
                            "suffix": item.suffix,
                            "path": str(item),
                            "modified": item.stat().st_mtime
                        })
            except Exception as e:
                files.append({"error": str(e)})
                
            result["directories"][topic] = {
                "exists": True,
                "item_count": len(files),
                "items": files
            }
        else:
            result["directories"][topic] = {"exists": False}
    
    return result

@app.post("/debug/recreate-service")
async def recreate_service(authenticated: bool = Depends(verify_auth)):
    """Recreate the auto-ingestion service from scratch"""
    global auto_ingestion_service
    
    try:
        # Stop existing service if it exists
        if auto_ingestion_service:
            if hasattr(auto_ingestion_service, 'observer') and auto_ingestion_service.observer:
                try:
                    if auto_ingestion_service.observer.is_alive():
                        auto_ingestion_service.observer.stop()
                        auto_ingestion_service.observer.join()
                    logger.info("🛑 Stopped existing observer")
                except Exception as e:
                    logger.warning(f"⚠️ Error stopping observer: {e}")
            
            auto_ingestion_service = None
            logger.info("🗑️ Cleared existing service")
        
        # Recreate service
        if not AutoIngestionService:
            return {"error": "AutoIngestionService class not available"}
        
        logger.info("🔧 Creating new auto-ingestion service...")
        auto_ingestion_service = AutoIngestionService()
        logger.info("✅ New auto-ingestion service created")
        
        # Check what methods are available
        available_methods = [method for method in dir(auto_ingestion_service) if not method.startswith('_')]
        
        return {
            "message": "Auto-ingestion service recreated",
            "available_methods": available_methods,
            "has_get_status": hasattr(auto_ingestion_service, 'get_status'),
            "has_queue": hasattr(auto_ingestion_service, 'queue'),
            "has_observer": hasattr(auto_ingestion_service, 'observer')
        }
        
    except Exception as e:
        logger.error(f"❌ Recreate service error: {e}")
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.post("/debug/manual-scan")
async def manual_scan(authenticated: bool = Depends(verify_auth)):
    """Manually scan for files and add to queue"""
    if not auto_ingestion_service:
        return {"error": "Auto-ingestion service not available"}
    
    try:
        # Use the correct method name
        found_files = await auto_ingestion_service.scan_existing_files()
        
        return {
            "message": f"Manual scan completed - found {found_files} files",
            "queue_status": auto_ingestion_service.queue.get_status(),
            "service_status": auto_ingestion_service.get_status()
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.post("/debug/start-file-watcher")
async def start_file_watcher(authenticated: bool = Depends(verify_auth)):
    """Manually start the file watcher"""
    if not auto_ingestion_service:
        return {"error": "Auto-ingestion service not available"}
    
    try:
        if auto_ingestion_service.observer and auto_ingestion_service.handler:
            if not auto_ingestion_service.observer.is_alive():
                auto_ingestion_service.observer.schedule(
                    auto_ingestion_service.handler,
                    str(auto_ingestion_service.watch_dir),
                    recursive=True
                )
                auto_ingestion_service.observer.start()
                
                # Wait a moment for it to start
                await asyncio.sleep(1)
                
                return {
                    "message": "File watcher started",
                    "observer_alive": auto_ingestion_service.observer.is_alive(),
                    "watched_path": str(auto_ingestion_service.watch_dir)
                }
            else:
                return {"message": "File watcher already running"}
        else:
            return {"error": "File watcher components not available"}
            
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.post("/debug/detailed-scan")
async def detailed_scan(authenticated: bool = Depends(verify_auth)):
    """Detailed scan with debugging information"""
    if not auto_ingestion_service:
        return {"error": "Auto-ingestion service not available"}
    
    watch_dir = Path("/app/ingestion-watch")
    topic_folders = ['business-central', 'ai-research', 'cloud-computing', 'general']
    
    # Get supported extensions from the service (not hardcoded)
    if hasattr(auto_ingestion_service, 'supported_extensions'):
        supported_extensions = auto_ingestion_service.supported_extensions
    else:
        # Fallback to extended list including RTF
        supported_extensions = ['.md', '.txt', '.pdf', '.docx', '.rtf']
    
    scan_results = {
        "watch_dir_exists": watch_dir.exists(),
        "watch_dir_path": str(watch_dir),
        "supported_extensions": supported_extensions,
        "scan_details": {},
        "total_files_found": 0,
        "total_files_queued": 0
    }
    
    try:
        for topic in topic_folders:
            topic_dir = watch_dir / topic
            topic_info = {
                "directory_exists": topic_dir.exists(),
                "directory_path": str(topic_dir),
                "items_found": [],
                "files_processed": 0,
                "files_skipped": 0,
                "skip_reasons": []
            }
            
            if topic_dir.exists():
                try:
                    # List all items in directory
                    for item in topic_dir.iterdir():
                        item_info = {
                            "name": item.name,
                            "path": str(item),
                            "is_file": item.is_file(),
                            "is_dir": item.is_dir(),
                        }
                        
                        if item.is_file():
                            item_info.update({
                                "size": item.stat().st_size,
                                "suffix": item.suffix,
                                "suffix_supported": item.suffix in supported_extensions
                            })
                            
                            # Check if this file would be processed
                            if item.suffix in supported_extensions:
                                # Try to queue this file
                                try:
                                    job_id = await auto_ingestion_service.queue.add_job(str(item), topic, priority=1)
                                    item_info["queued"] = True
                                    item_info["job_id"] = job_id
                                    topic_info["files_processed"] += 1
                                    scan_results["total_files_queued"] += 1
                                except Exception as e:
                                    item_info["queue_error"] = str(e)
                                    topic_info["files_skipped"] += 1
                                    topic_info["skip_reasons"].append(f"{item.name}: {str(e)}")
                            else:
                                topic_info["files_skipped"] += 1
                                topic_info["skip_reasons"].append(f"{item.name}: unsupported extension '{item.suffix}'")
                            
                            scan_results["total_files_found"] += 1
                        
                        topic_info["items_found"].append(item_info)
                        
                except Exception as e:
                    topic_info["scan_error"] = str(e)
            
            scan_results["scan_details"][topic] = topic_info
        
        # Get final queue status
        scan_results["final_queue_status"] = auto_ingestion_service.queue.get_status()
        
        return scan_results
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.post("/debug/list-all-files")
async def list_all_files(authenticated: bool = Depends(verify_auth)):
    """List all files in watch directory with full details"""
    watch_dir = Path("/app/ingestion-watch")
    
    def scan_directory(directory: Path, max_depth: int = 3, current_depth: int = 0):
        items = []
        if current_depth >= max_depth:
            return items
            
        try:
            for item in directory.iterdir():
                item_info = {
                    "name": item.name,
                    "path": str(item),
                    "relative_path": str(item.relative_to(watch_dir)),
                    "is_file": item.is_file(),
                    "is_dir": item.is_dir(),
                    "depth": current_depth
                }
                
                if item.is_file():
                    try:
                        stat = item.stat()
                        item_info.update({
                            "size": stat.st_size,
                            "suffix": item.suffix,
                            "modified": stat.st_mtime,
                            "permissions": oct(stat.st_mode)[-3:]
                        })
                    except Exception as e:
                        item_info["stat_error"] = str(e)
                
                items.append(item_info)
                
                if item.is_dir():
                    # Recursively scan subdirectories
                    subdirectory_items = scan_directory(item, max_depth, current_depth + 1)
                    items.extend(subdirectory_items)
                    
        except Exception as e:
            items.append({
                "error": f"Cannot scan directory {directory}: {str(e)}",
                "path": str(directory)
            })
        
        return items
    
    try:
        all_items = scan_directory(watch_dir)
        
        return {
            "watch_directory": str(watch_dir),
            "directory_exists": watch_dir.exists(),
            "total_items": len(all_items),
            "files": [item for item in all_items if item.get("is_file")],
            "directories": [item for item in all_items if item.get("is_dir")],
            "errors": [item for item in all_items if "error" in item],
            "all_items": all_items
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/debug/database-status")
async def database_status(authenticated: bool = Depends(verify_auth)):
    """Check what's actually in the database"""
    import asyncpg
    
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        
        # Check tables exist
        tables = await conn.fetch("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('topics', 'documents', 'chunks')
        """)
        
        # Check topics
        topics = await conn.fetch("SELECT * FROM topics")
        
        # Check documents
        documents = await conn.fetch("SELECT id, title, topic_id, created_at FROM documents LIMIT 10")
        
        # Check chunks
        chunks = await conn.fetch("SELECT COUNT(*) as count FROM chunks")
        
        await conn.close()
        
        return {
            "tables_exist": [dict(t) for t in tables],
            "topics": [dict(t) for t in topics],
            "documents_count": len(documents),
            "documents_sample": [dict(d) for d in documents],
            "chunks_count": dict(chunks[0])['count'] if chunks else 0
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.post("/debug/start-processing")
async def start_processing(authenticated: bool = Depends(verify_auth)):
    """Manually start the background processing"""
    if not auto_ingestion_service:
        return {"error": "Auto-ingestion service not available"}
    
    try:
        if hasattr(auto_ingestion_service, 'processing_task') and not auto_ingestion_service.processing_task.done():
            return {"message": "Background processing already running"}
        
        # Start the background processing
        logger.info("🚀 Manually starting background processing...")
        processing_task = asyncio.create_task(auto_ingestion_service.process_jobs_forever())
        auto_ingestion_service.processing_task = processing_task
        
        # Give it a moment to start
        await asyncio.sleep(0.5)
        
        return {
            "message": "Background processing started",
            "queue_status": auto_ingestion_service.queue.get_status(),
            "task_running": not processing_task.done()
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.post("/debug/process-single-file")
async def process_single_file(
    file_path: str,
    topic: str,
    authenticated: bool = Depends(verify_auth)
):
    """Process a single file manually"""
    if not auto_ingestion_service:
        return {"error": "Auto-ingestion service not available"}
    
    try:
        # Add file to queue
        job_id = await auto_ingestion_service.queue.add_job(file_path, topic, priority=10)
        
        return {
            "message": f"File queued: {file_path}",
            "job_id": job_id,
            "topic": topic,
            "queue_status": auto_ingestion_service.queue.get_status()
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


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

@app.get("/debug/version")
async def debug_version():
    """Check what version of code is running"""
    return {
        "message": "Updated version with auto-ingestion debugging",
        "timestamp": "2024-12-26-v3",
        "auto_ingestion_service_instance": auto_ingestion_service is not None,
        "auto_ingestion_class_available": AutoIngestionService is not None
    }

@app.get("/debug/lifespan-status")
async def debug_lifespan_status():
    """Check if lifespan was called and service state (safe version)"""
    result = {
        "auto_ingestion_service_created": auto_ingestion_service is not None,
        "auto_ingestion_class_available": AutoIngestionService is not None,
        "timestamp": "2024-12-26-v5"
    }
    
    if auto_ingestion_service:
        # Safely check for methods
        try:
            if hasattr(auto_ingestion_service, 'get_status'):
                result["service_status"] = auto_ingestion_service.get_status()
            else:
                result["service_status"] = "get_status method missing"
                
            # Check available methods
            result["available_methods"] = [method for method in dir(auto_ingestion_service) if not method.startswith('_')]
            
            # Check queue if available
            if hasattr(auto_ingestion_service, 'queue'):
                if hasattr(auto_ingestion_service.queue, 'get_status'):
                    result["queue_status"] = auto_ingestion_service.queue.get_status()
                else:
                    result["queue_status"] = "queue.get_status method missing"
            else:
                result["queue_status"] = "queue attribute missing"
                
        except Exception as e:
            result["service_status"] = f"Error checking status: {e}"
    else:
        result["service_status"] = "No service instance"
    
    return result

@app.post("/debug/force-create-service")
async def force_create_service(authenticated: bool = Depends(verify_auth)):
    """Force create the auto-ingestion service"""
    global auto_ingestion_service
    
    try:
        if auto_ingestion_service:
            return {"message": "Service already exists", "status": auto_ingestion_service.get_status()}
        
        logger.info("🔧 FORCE: Creating auto-ingestion service...")
        auto_ingestion_service = AutoIngestionService()
        logger.info("✅ FORCE: Service created successfully")
        
        return {
            "message": "Auto-ingestion service created successfully",
            "status": auto_ingestion_service.get_status(),
            "created_at": "force-created"
        }
        
    except Exception as e:
        logger.error(f"❌ FORCE: Failed to create service: {e}")
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


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

@app.post("/debug/create-topics")
async def create_topics(authenticated: bool = Depends(verify_auth)):
    """Create required topics in database"""
    import asyncpg
    
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        
        # Create topics table if it doesn't exist
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name VARCHAR(255) NOT NULL UNIQUE,
                description TEXT,
                slug VARCHAR(100) NOT NULL UNIQUE,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert default topics
        topics_to_create = [
            ('Microsoft Business Central', 'ERP and business management system content', 'business-central'),
            ('AI Research', 'Artificial Intelligence research and developments', 'ai-research'),
            ('Cloud Computing', 'Cloud services and infrastructure', 'cloud-computing'),
            ('General', 'General purpose content', 'general')
        ]
        
        created_topics = []
        for name, description, slug in topics_to_create:
            try:
                topic_id = await conn.fetchval("""
                    INSERT INTO topics (name, description, slug) 
                    VALUES ($1, $2, $3) 
                    ON CONFLICT (slug) DO UPDATE SET 
                        name = EXCLUDED.name,
                        description = EXCLUDED.description
                    RETURNING id
                """, name, description, slug)
                created_topics.append({"name": name, "slug": slug, "id": str(topic_id)})
            except Exception as e:
                created_topics.append({"name": name, "slug": slug, "error": str(e)})
        
        await conn.close()
        
        return {
            "message": "Topics created/updated",
            "topics": created_topics
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.post("/debug/test-topic-ingestion")
async def test_topic_ingestion(authenticated: bool = Depends(verify_auth)):
    """Test topic ingestion directly"""
    try:
        if not auto_ingestion_service:
            return {"error": "Auto-ingestion service not available"}
        
        # Create a simple test file
        test_dir = Path("/tmp/topic_test")
        test_dir.mkdir(exist_ok=True)
        
        test_file = test_dir / "test_document.md"
        test_file.write_text("""
# Test Document

This is a test document to verify that topic ingestion is working correctly.

## Section 1
Some content about Microsoft Dynamics 365 Business Central.

## Section 2  
More content to test the chunking process.
""")
        
        # Test ingestion
        logger.info("🧪 Testing direct topic ingestion...")
        
        try:
            await auto_ingestion_service.topic_ingester.ingest_topic_documents(
                topic_slug="business-central",
                documents_path=str(test_dir),
                clean=False
            )
            
            # Cleanup
            shutil.rmtree(test_dir)
            
            return {
                "result": "SUCCESS",
                "message": "Direct topic ingestion completed without errors"
            }
            
        except Exception as ingestion_error:
            # Cleanup
            shutil.rmtree(test_dir)
            
            import traceback
            return {
                "result": "FAILED",
                "error": str(ingestion_error),
                "traceback": traceback.format_exc()
            }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

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
