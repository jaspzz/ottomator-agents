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

# File watcher imports
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object
    Observer = None

from .topic_ingest import TopicIngestion

logger = logging.getLogger(__name__)

class IngestionQueue:
    """Thread-safe ingestion queue"""
    def __init__(self):
        self.queue = asyncio.Queue()
        self.processing = {}
        self.completed = {}
        self.failed = {}
        self._lock = threading.Lock()
        logger.info("🔧 IngestionQueue initialized")
    
    def add_job_sync(self, file_path: str, topic: str, priority: int = 1) -> str:
        """Add a job to ingestion queue (thread-safe)"""
        job_id = hashlib.md5(f"{file_path}_{topic}_{datetime.now().timestamp()}".encode()).hexdigest()[:8]
        job = {
            'id': job_id,
            'file_path': file_path,
            'topic': topic,
            'priority': priority,
            'created_at': datetime.now().isoformat(),
            'status': 'queued'
        }
        
        # Schedule the async operation
        try:
            # Get the running event loop
            loop = asyncio.get_running_loop()
            # Schedule the coroutine to run in the event loop
            asyncio.run_coroutine_threadsafe(self.queue.put(job), loop)
            logger.info(f"📋 Queued job {job_id}: {Path(file_path).name} → {topic}")
        except RuntimeError:
            # No running event loop, store for later
            with self._lock:
                if not hasattr(self, '_pending_jobs'):
                    self._pending_jobs = []
                self._pending_jobs.append(job)
            logger.info(f"📋 Pending job {job_id}: {Path(file_path).name} → {topic}")
        
        return job_id
    
    async def process_pending_jobs(self):
        """Process any jobs that were queued before the event loop started"""
        if hasattr(self, '_pending_jobs'):
            with self._lock:
                pending_count = len(getattr(self, '_pending_jobs', []))
                if pending_count > 0:
                    for job in self._pending_jobs:
                        await self.queue.put(job)
                        logger.info(f"📋 Processed pending job: {job['id']}")
                    self._pending_jobs = []
                    logger.info(f"✅ Processed {pending_count} pending jobs")
    
    async def add_job(self, file_path: str, topic: str, priority: int = 1):
        """Add a job to ingestion queue (async)"""
        return self.add_job_sync(file_path, topic, priority)
    
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

class AutoIngestionHandler(FileSystemEventHandler):
    """Handle file system events for auto-ingestion"""
    
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.watch_dir = Path("/app/ingestion-watch")
        self.supported_extensions = {'.md', '.txt', '.pdf', '.docx'}
        
        # Topic mapping from folder names
        self.topic_mapping = {
            'business-central': 'business-central',
            'ai-research': 'ai-research', 
            'cloud-computing': 'cloud-computing',
            'general': 'general'
        }
        
        logger.info(f"🔧 File handler initialized, watching: {self.watch_dir}")
    
    def on_created(self, event):
        """Handle file creation"""
        if not event.is_directory:
            logger.info(f"📁 File created: {event.src_path}")
            self.process_new_file_sync(event.src_path)
    
    def on_moved(self, event):
        """Handle file moves (SFTP often moves files)"""
        if not event.is_directory:
            logger.info(f"📁 File moved: {event.dest_path}")
            self.process_new_file_sync(event.dest_path)
    
    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory:
            logger.info(f"📁 File modified: {event.src_path}")
            # Wait a bit for file to be fully written, then process
            threading.Timer(3.0, lambda: self.process_new_file_sync(event.src_path)).start()
    
    def process_new_file_sync(self, file_path: str):
        """Process newly detected file (synchronous version for watchdog)"""
        try:
            path = Path(file_path)
            logger.info(f"🔍 Processing new file: {path.name}")
            
            # Skip hidden files, temp files, and unsupported formats
            if (path.name.startswith('.') or 
                path.name.startswith('~') or 
                path.name.startswith('#') or
                path.suffix not in self.supported_extensions):
                logger.info(f"⏭️ Skipping unsupported file: {path.name}")
                return
            
            # Determine topic from folder structure
            try:
                relative_path = path.relative_to(self.watch_dir)
                topic_folder = relative_path.parts[0] if relative_path.parts else 'general'
                logger.info(f"📂 File in folder: {topic_folder}")
            except ValueError:
                logger.warning(f"⚠️ File not in watch directory: {path}")
                return
                
            topic = self.topic_mapping.get(topic_folder, 'general')
            
            # Skip if file is in processed/failed folders
            if topic_folder in ['processed', 'failed']:
                logger.info(f"⏭️ Skipping file in {topic_folder} folder")
                return
            
            # Check if file is stable (not still being uploaded)
            if not self.is_file_stable_sync(file_path):
                logger.info(f"⏳ File still uploading, will retry: {path.name}")
                # Schedule retry
                threading.Timer(5.0, lambda: self.process_new_file_sync(file_path)).start()
                return
            
            # Add to ingestion queue (thread-safe)
            job_id = self.queue.add_job_sync(str(path), topic)
            logger.info(f"✅ Queued file: {path.name} → Topic: {topic} → Job: {job_id}")
            
        except Exception as e:
            logger.error(f"❌ Error processing new file {file_path}: {e}")
    
    def is_file_stable_sync(self, file_path: str, wait_time: int = 2) -> bool:
        """Check if file size is stable (synchronous version)"""
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            
            size1 = path.stat().st_size
            import time
            time.sleep(wait_time)
            
            if not path.exists():
                return False
                
            size2 = path.stat().st_size
            
            is_stable = size1 == size2 and size1 > 0
            logger.info(f"📊 File stability check for {path.name}: {size1} → {size2} bytes, stable: {is_stable}")
            return is_stable
            
        except Exception as e:
            logger.error(f"❌ Error checking file stability: {e}")
            return False

class AutoIngestionService:
    """Auto-ingestion service with file watcher and processing queue"""
    
    def __init__(self):
        logger.info("🔧 Initializing AutoIngestionService...")
        
        # Initialize queue first
        self.queue = IngestionQueue()
        logger.info("📋 Queue initialized")
        
        # Initialize topic ingester
        self.topic_ingester = TopicIngestion()
        logger.info("📚 Topic ingester initialized")
        
        # Set up directories
        self.watch_dir = Path("/app/ingestion-watch")
        self.processed_dir = self.watch_dir / "processed"
        self.failed_dir = self.watch_dir / "failed"
        
        # Create directories
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Directories created: {self.watch_dir}")
        
        # File watcher components (only if watchdog is available)
        if WATCHDOG_AVAILABLE:
            self.handler = AutoIngestionHandler(self.queue)
            self.observer = Observer()
            logger.info("👁️ File watcher components initialized")
        else:
            self.handler = None
            self.observer = None
            logger.warning("⚠️ Watchdog not available, file watching disabled")
        
        logger.info("✅ AutoIngestionService initialization complete")

    def get_status(self) -> Dict:
        try:
            return {
                'service': 'auto-ingestion-service',
                'status': 'running',
                'watched_directory': str(self.watch_dir),
                'watchdog_available': WATCHDOG_AVAILABLE,
                'observer_alive': self.observer.is_alive() if self.observer else False,
                'queue_status': self.queue.get_status() if hasattr(self.queue, 'get_status') else "queue status unavailable",
                'recent_completed': list(self.queue.completed.values())[-5:] if hasattr(self.queue, 'completed') else [],
                'recent_failed': list(self.queue.failed.values())[-5:] if hasattr(self.queue, 'failed') else []
            }
        except Exception as e:
            return {
                'service': 'auto-ingestion-service',
                'status': 'error',
                'error': str(e)
            }
    
    async def start(self):
        """Start the auto-ingestion service"""
        logger.info("🚀 Starting auto-ingestion service...")
        
        # Process any pending jobs first
        await self.queue.process_pending_jobs()
        
        # Start file watcher if available
        if self.observer and self.handler:
            try:
                self.observer.schedule(
                    self.handler,
                    str(self.watch_dir),
                    recursive=True
                )
                self.observer.start()
                logger.info(f"👁️ File watcher started, monitoring: {self.watch_dir}")
                logger.info(f"👁️ Observer alive: {self.observer.is_alive()}")
            except Exception as e:
                logger.error(f"❌ Failed to start file watcher: {e}")
        else:
            logger.info("📝 File watcher not available, using manual scanning only")
        
        # Process existing files
        await self.scan_existing_files()
        
        # Start background processor - THIS IS THE KEY FIX
        logger.info("⚙️ Starting background job processor...")
        
        # Create the processing task but don't await it (so it runs in background)
        processing_task = asyncio.create_task(self.process_jobs_loop())
        logger.info("📋 Background processing task created")
        
        # Don't await the processing_task here - let it run in background
        # The start() method should return so the lifespan can complete
    
async def process_jobs_loop(self):
    """Main processing loop - runs forever in background"""
    logger.info("🔄 Starting job processing loop...")
    
    while True:
        try:
            logger.info("⏳ Waiting for jobs in queue...")
            # Get next job from queue
            job = await self.queue.get_job()
            self.queue.processing[job['id']] = job
            
            logger.info(f"⚙️ Processing job {job['id']}: {Path(job['file_path']).name}")
            
            # Process the file
            success = await self.process_job(job)
            
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
            await asyncio.sleep(5)  # Wait before continuing

    
    async def scan_existing_files(self):
        """Scan for existing files in watch directories"""
        logger.info("🔍 Scanning for existing files...")
        
        topic_folders = ['business-central', 'ai-research', 'cloud-computing', 'general']
        found_files = 0
        
        for topic in topic_folders:
            topic_dir = self.watch_dir / topic
            if topic_dir.exists():
                logger.info(f"📂 Scanning {topic} directory...")
                for file_path in topic_dir.glob("*"):
                    if file_path.is_file() and file_path.suffix in ['.md', '.txt', '.pdf', '.docx']:
                        job_id = await self.queue.add_job(str(file_path), topic, priority=1)
                        found_files += 1
                        logger.info(f"📋 Queued existing file: {file_path.name} → {topic} → Job: {job_id}")
                    else:
                        logger.info(f"⏭️ Skipped: {file_path.name} (unsupported or not a file)")
            else:
                logger.info(f"📂 Creating topic directory: {topic_dir}")
                topic_dir.mkdir(exist_ok=True)
        
        if found_files > 0:
            logger.info(f"✅ Found and queued {found_files} existing files")
        else:
            logger.info("📭 No existing files found")
    
    async def process_job(self, job: Dict) -> bool:
        """Process a single ingestion job"""
        try:
            file_path = Path(job['file_path'])
            topic = job['topic']
            
            if not file_path.exists():
                logger.error(f"❌ File not found: {file_path}")
                return False
            
            logger.info(f"📄 Processing file: {file_path.name} ({file_path.stat().st_size} bytes)")
            
            # Create temporary directory for this file
            temp_dir = Path(f"/tmp/ingestion_{job['id']}")
            temp_dir.mkdir(exist_ok=True)
            
            # Copy file to temp directory
            temp_file = temp_dir / file_path.name
            shutil.copy2(file_path, temp_file)
            
            # Process with topic ingester
            await self.topic_ingester.ingest_topic_documents(
                topic_slug=topic,
                documents_path=str(temp_dir),
                clean=False
            )
            
            # Cleanup temp directory
            shutil.rmtree(temp_dir)
            
            logger.info(f"✅ Successfully processed: {file_path.name} → {topic}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to process job {job['id']}: {e}")
            await self.save_error_log(job, str(e))
            return False
    
    async def move_file_to_processed(self, job: Dict):
        """Move successfully processed file"""
        try:
            source = Path(job['file_path'])
            if source.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                destination = self.processed_dir / f"{timestamp}_{source.name}"
                shutil.move(str(source), str(destination))
                logger.info(f"📦 Moved to processed: {destination.name}")
                
                # Create success log
                log_file = destination.with_suffix('.log')
                log_file.write_text(json.dumps({
                    'job_id': job['id'],
                    'original_path': job['file_path'],
                    'topic': job['topic'],
                    'processed_at': datetime.now().isoformat(),
                    'status': 'success'
                }, indent=2))
            
        except Exception as e:
            logger.error(f"❌ Failed to move processed file: {e}")
    
    async def move_file_to_failed(self, job: Dict):
        """Move failed file"""
        try:
            source = Path(job['file_path'])
            if source.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                destination = self.failed_dir / f"{timestamp}_{source.name}"
                shutil.move(str(source), str(destination))
                logger.info(f"📦 Moved to failed: {destination.name}")
        except Exception as e:
            logger.error(f"❌ Failed to move failed file: {e}")
    
    async def save_error_log(self, job: Dict, error: str):
        """Save error details"""
        try:
            error_log = self.failed_dir / f"error_{job['id']}.log"
            error_log.write_text(json.dumps({
                'job_id': job['id'],
                'file_path': job['file_path'],
                'topic': job['topic'],
                'error': error,
                'failed_at': datetime.now().isoformat()
            }, indent=2))
        except Exception as e:
            logger.error(f"❌ Failed to save error log: {e}")
    
    def get_status(self) -> Dict:
        """Get service status"""
        return {
            'service': 'auto-ingestion-service',
            'status': 'running',
            'watched_directory': str(self.watch_dir),
            'watchdog_available': WATCHDOG_AVAILABLE,
            'observer_alive': self.observer.is_alive() if self.observer else False,
            'queue_status': self.queue.get_status(),
            'recent_completed': list(self.queue.completed.values())[-5:],
            'recent_failed': list(self.queue.failed.values())[-5:]
        }
