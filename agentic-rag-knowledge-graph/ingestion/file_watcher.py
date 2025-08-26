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
    """Auto-ingestion service - complete working version"""
    
    def __init__(self):
        logger.info("🔧 Initializing AutoIngestionService...")
        
        # Initialize components
        self.queue = IngestionQueue()
        self.topic_ingester = TopicIngestion()
        
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
        
        # Start processing loop
        logger.info("⚙️ Starting processing loop...")
        await self.process_jobs_forever()
    
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
                        logger.info(f"📋 Queued existing file: {file_path.name} → {topic}")
                    else:
                        if file_path.is_file():
                            logger.info(f"⏭️ Skipped unsupported file: {file_path.name}")
            else:
                logger.info(f"📁 Creating topic directory: {topic_dir}")
                topic_dir.mkdir(exist_ok=True)
        
        logger.info(f"✅ Scan complete: found {found_files} files")
        return found_files
    
    async def process_jobs_forever(self):
        """Process jobs forever"""
        logger.info("🔄 Starting job processing loop...")
        
        while True:
            try:
                logger.info("⏳ Waiting for jobs...")
                
                # Get next job
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
        """Process a single job"""
        try:
            file_path = Path(job['file_path'])
            topic = job['topic']
            
            if not file_path.exists():
                logger.error(f"❌ File not found: {file_path}")
                return False
            
            logger.info(f"📄 Processing: {file_path.name} ({file_path.stat().st_size} bytes)")
            
            # Create temp directory
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
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
            logger.info(f"✅ Successfully processed: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to process job {job['id']}: {e}")
            return False
    
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
            'watchdog_available': WATCHDOG_AVAILABLE,
            'queue_status': self.queue.get_status(),
            'recent_completed': list(self.queue.completed.values())[-5:],
            'recent_failed': list(self.queue.failed.values())[-5:]
        }
