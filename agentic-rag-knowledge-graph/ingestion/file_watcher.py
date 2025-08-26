import asyncio
import logging
import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import hashlib
import threading

# File watcher imports (optional)
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object

from .topic_ingest import TopicIngestion

logger = logging.getLogger(__name__)

class IngestionQueue:
    """Simple ingestion queue"""
    def __init__(self):
        self.queue = asyncio.Queue()
        self.processing = {}
        self.completed = {}
        self.failed = {}
        logger.info("📋 IngestionQueue initialized")
    
    async def add_job(self, file_path: str, topic: str, priority: int = 1):
        """Add a job to ingestion queue"""
        job_id = hashlib.md5(f"{file_path}_{topic}_{datetime.now().timestamp()}".encode()).hexdigest()[:8]
        job = {
            'id': job_id,
            'file_path': file_path,
            'topic': topic,
            'priority': priority,
            'created_at': datetime.now().isoformat(),
            'status': 'queued'
        }
        
        await self.queue.put(job)
        logger.info(f"📋 Queued job {job_id}: {Path(file_path).name} → {topic}")
        return job_id
    
    async def get_job(self):
        """Get next job from queue"""
        return await self.queue.get()
    
    def get_status(self) -> Dict:
        """Get queue status"""
        return {
            'queued': self.queue.qsize(),
            'processing': len(self.processing),
            'completed': len(self.completed),
            'failed': len(self.failed)
        }

class AutoIngestionService:
    
    def __init__(self):
        logger.info("🔧 Initializing AutoIngestionService...")
        
        # Initialize components
        self.queue = IngestionQueue()
        self.topic_ingester = TopicIngestion()
        
        # Set supported extensions including RTF
        self.supported_extensions = ['.md', '.txt', '.pdf', '.docx', '.rtf']
        logger.info(f"📋 Supported extensions: {self.supported_extensions}")
        
        # Set up directories
        self.watch_dir = Path("/app/ingestion-watch")
        self.processed_dir = self.watch_dir / "processed"
        self.failed_dir = self.watch_dir / "failed"
        
        # Create directories
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        
        # File watcher (if available)
        self.observer = Observer() if WATCHDOG_AVAILABLE else None
        self.handler = None
        
        logger.info("✅ AutoIngestionService initialization complete")
    
    async def start(self):
        """Start the auto-ingestion service"""
        logger.info("🚀 Starting auto-ingestion service...")
        
        # Scan for existing files
        await self.scan_existing_files()
        
        # Start processing loop in background (don't await it!)
        logger.info("⚙️ Starting background processing task...")
        
        # Create the processing task but DON'T await it
        processing_task = asyncio.create_task(self.process_jobs_forever())
        logger.info("📋 Background processing task created and running")
        
        # Store the task so we can stop it later if needed
        self.processing_task = processing_task
        
        logger.info("✅ Auto-ingestion service start complete")
        # Return here so the lifespan can continue
        
    async def scan_existing_files(self):
        """Scan for existing files in watch directories"""
        logger.info("🔍 Scanning for existing files...")
        logger.info(f"📋 Looking for files with extensions: {self.supported_extensions}")
        
        topic_folders = ['business-central', 'ai-research', 'cloud-computing', 'general']
        found_files = 0
        
        for topic in topic_folders:
            topic_dir = self.watch_dir / topic
            if topic_dir.exists():
                logger.info(f"📂 Scanning {topic} directory...")
                for file_path in topic_dir.glob("*"):
                    if file_path.is_file():
                        if file_path.suffix in self.supported_extensions:
                            job_id = await self.queue.add_job(str(file_path), topic, priority=1)
                            found_files += 1
                            logger.info(f"📋 Queued existing file: {file_path.name} → {topic}")
                        else:
                            logger.info(f"⏭️ Skipped unsupported file: {file_path.name} (extension: {file_path.suffix})")
                    else:
                        logger.info(f"⏭️ Skipped non-file: {file_path.name}")
            else:
                logger.info(f"📁 Creating topic directory: {topic_dir}")
                topic_dir.mkdir(exist_ok=True)
        
        logger.info(f"✅ Scan complete: found {found_files} files")
        return found_files
    
    async def process_jobs_forever(self):
        """Process jobs forever in background"""
        logger.info("🔄 Background job processor starting...")
        
        while True:
            try:
                logger.info("⏳ Waiting for jobs in queue...")
                
                # Get next job from queue
                job = await self.queue.get_job()
                self.queue.processing[job['id']] = job
                
                logger.info(f"⚙️ Processing job {job['id']}: {Path(job['file_path']).name}")
                
                # Process the job
                success = await self.process_single_job(job)
                
                # Move job to completed or failed
                if success:
                    self.queue.completed[job['id']] = {**job, 'completed_at': datetime.now().isoformat()}
                    await self.move_file_to_processed(job)
                    logger.info(f"✅ Job {job['id']} completed successfully")
                else:
                    self.queue.failed[job['id']] = {**job, 'failed_at': datetime.now().isoformat()}
                    await self.move_file_to_failed(job)
                    logger.error(f"❌ Job {job['id']} failed")
                
                # Remove from processing
                del self.queue.processing[job['id']]
                
            except Exception as e:
                logger.error(f"❌ Processing loop error: {e}")
                await asyncio.sleep(5)
    
    async def process_single_job(self, job: Dict) -> bool:
        """Process a single job with detailed error handling"""
        try:
            file_path = Path(job['file_path'])
            topic = job['topic']
            
            logger.info(f"📄 Starting job processing: {file_path.name}")
            
            if not file_path.exists():
                logger.error(f"❌ File not found: {file_path}")
                return False
            
            logger.info(f"📊 File details: {file_path.name} ({file_path.stat().st_size} bytes, {file_path.suffix})")
            
            # Create temp directory
            temp_dir = Path(f"/tmp/ingestion_{job['id']}")
            temp_dir.mkdir(exist_ok=True)
            logger.info(f"📁 Created temp directory: {temp_dir}")
            
            # Copy file to temp directory
            temp_file = temp_dir / file_path.name
            shutil.copy2(file_path, temp_file)
            logger.info(f"📋 Copied file to temp: {temp_file}")
            
            # Process with topic ingester (with detailed error handling)
            try:
                logger.info(f"🚀 Starting topic ingestion for topic: {topic}")
                await self.topic_ingester.ingest_topic_documents(
                    topic_slug=topic,
                    documents_path=str(temp_dir),
                    clean=False
                )
                logger.info(f"✅ Topic ingestion completed successfully")
            except Exception as ingestion_error:
                logger.error(f"❌ Topic ingestion failed: {ingestion_error}")
                import traceback
                logger.error(f"🔍 Ingestion traceback: {traceback.format_exc()}")
                
                # Save error details but don't fail completely
                error_log = {
                    "job_id": job['id'],
                    "file_path": job['file_path'],
                    "topic": topic,
                    "ingestion_error": str(ingestion_error),
                    "traceback": traceback.format_exc(),
                    "failed_at": datetime.now().isoformat()
                }
                
                error_file = self.failed_dir / f"ingestion_error_{job['id']}.json"
                error_file.write_text(json.dumps(error_log, indent=2))
                
                # Cleanup temp directory
                shutil.rmtree(temp_dir)
                return False
            
            # Cleanup temp directory
            shutil.rmtree(temp_dir)
            logger.info(f"🧹 Cleaned up temp directory")
            
            logger.info(f"✅ Job processing completed successfully: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Critical job processing error: {e}")
            import traceback
            logger.error(f"🔍 Critical traceback: {traceback.format_exc()}")
            return False
    

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

    async def move_file_to_processed(self, job: Dict):
        """Move file to processed directory"""
        try:
            source = Path(job['file_path'])
            if source.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                destination = self.processed_dir / f"{timestamp}_{source.name}"
                shutil.move(str(source), str(destination))
                logger.info(f"📦 Moved to processed: {destination.name}")
        except Exception as e:
            logger.error(f"❌ Failed to move processed file: {e}")
    
    async def move_file_to_failed(self, job: Dict):
        """Move file to failed directory"""
        try:
            source = Path(job['file_path'])
            if source.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                destination = self.failed_dir / f"{timestamp}_{source.name}"
                shutil.move(str(source), str(destination))
                logger.info(f"📦 Moved to failed: {destination.name}")
        except Exception as e:
            logger.error(f"❌ Failed to move failed file: {e}")
    
    def get_status(self) -> Dict:
        """Get service status"""
        return {
            'service': 'auto-ingestion-service',
            'status': 'running',
            'watched_directory': str(self.watch_dir),
            'supported_extensions': self.supported_extensions,
            'watchdog_available': WATCHDOG_AVAILABLE,
            'background_processor_running': hasattr(self, 'processing_task') and not self.processing_task.done(),
            'queue_status': self.queue.get_status(),
            'recent_completed': list(self.queue.completed.values())[-5:],
            'recent_failed': list(self.queue.failed.values())[-5:]
        }
