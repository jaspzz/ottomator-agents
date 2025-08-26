import asyncio
import logging
import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
import asyncpg
from datetime import datetime
import hashlib

from .topic_ingest import TopicIngestion

logger = logging.getLogger(__name__)

class IngestionQueue:
    """Simple in-memory queue for ingestion jobs"""
    def __init__(self):
        self.queue = asyncio.Queue()
        self.processing = {}
        self.completed = {}
        self.failed = {}
    
    async def add_job(self, file_path: str, topic: str, priority: int = 1):
        """Add a file to ingestion queue"""
        job_id = hashlib.md5(f"{file_path}_{topic}".encode()).hexdigest()[:8]
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

class AutoIngestionHandler(FileSystemEventHandler):
    """Handle file system events for auto-ingestion"""
    
    def __init__(self, queue: IngestionQueue):
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
    
    def on_created(self, event):
        """Handle file creation"""
        if not event.is_directory:
            asyncio.create_task(self.process_new_file(event.src_path))
    
    def on_modified(self, event):
        """Handle file modification (in case of large uploads)"""
        if not event.is_directory:
            # Wait a bit to ensure file is fully written
            asyncio.create_task(self.delayed_process(event.src_path))
    
    async def delayed_process(self, file_path: str):
        """Process file after a delay to ensure it's fully uploaded"""
        await asyncio.sleep(2)  # Wait 2 seconds
        await self.process_new_file(file_path)
    
    async def process_new_file(self, file_path: str):
        """Process newly detected file"""
        try:
            path = Path(file_path)
            
            # Skip hidden files, temp files, and unsupported formats
            if (path.name.startswith('.') or 
                path.name.startswith('~') or 
                path.suffix not in self.supported_extensions):
                return
            
            # Determine topic from folder structure
            relative_path = path.relative_to(self.watch_dir)
            topic_folder = relative_path.parts[0] if relative_path.parts else 'general'
            topic = self.topic_mapping.get(topic_folder, 'general')
            
            # Skip if file is in processed/failed folders
            if topic_folder in ['processed', 'failed']:
                return
            
            # Check if file is stable (not still being uploaded)
            if not await self.is_file_stable(file_path):
                logger.info(f"⏳ File still uploading: {path.name}")
                return
            
            # Add to ingestion queue
            job_id = await self.queue.add_job(str(path), topic)
            logger.info(f"📁 New file detected: {path.name} → Topic: {topic} → Job: {job_id}")
            
        except Exception as e:
            logger.error(f"❌ Error processing new file {file_path}: {e}")
    
    async def is_file_stable(self, file_path: str, wait_time: int = 1) -> bool:
        """Check if file size is stable (upload complete)"""
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            
            size1 = path.stat().st_size
            await asyncio.sleep(wait_time)
            size2 = path.stat().st_size
            
            return size1 == size2 and size1 > 0
        except Exception:
            return False

class AutoIngestionProcessor:
    """Background processor for ingestion jobs"""
    
    def __init__(self, queue: IngestionQueue):
        self.queue = queue
        self.topic_ingester = TopicIngestion()
        self.watch_dir = Path("/app/ingestion-watch")
        self.processed_dir = self.watch_dir / "processed"
        self.failed_dir = self.watch_dir / "failed"
        
        # Create directories
        self.processed_dir.mkdir(exist_ok=True)
        self.failed_dir.mkdir(exist_ok=True)
    
    async def start_processing(self):
        """Start the background processing loop"""
        logger.info("🚀 Starting auto-ingestion processor")
        
        while True:
            try:
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
                else:
                    self.queue.failed[job['id']] = {**job, 'failed_at': datetime.now().isoformat()}
                    await self.move_file_to_failed(job)
                
                # Remove from processing
                del self.queue.processing[job['id']]
                
            except Exception as e:
                logger.error(f"❌ Processing error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def process_job(self, job: Dict) -> bool:
        """Process a single ingestion job"""
        try:
            file_path = Path(job['file_path'])
            topic = job['topic']
            
            if not file_path.exists():
                logger.error(f"❌ File not found: {file_path}")
                return False
            
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
                clean=False  # Don't clean, just add
            )
            
            # Cleanup temp directory
            shutil.rmtree(temp_dir)
            
            logger.info(f"✅ Successfully processed: {file_path.name} → {topic}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to process job {job['id']}: {e}")
            # Save error details
            await self.save_error_log(job, str(e))
            return False
    
    async def move_file_to_processed(self, job: Dict):
        """Move successfully processed file"""
        try:
            source = Path(job['file_path'])
            destination = self.processed_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{source.name}"
            shutil.move(str(source), str(destination))
            
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
                destination = self.failed_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{source.name}"
                shutil.move(str(source), str(destination))
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

class AutoIngestionService:
    """Main service that coordinates file watching and processing"""
    
    def __init__(self):
        self.queue = IngestionQueue()
        self.handler = AutoIngestionHandler(self.queue)
        self.processor = AutoIngestionProcessor(self.queue)
        self.observer = Observer()
        self.watch_dir = "/app/ingestion-watch"
    
    async def start(self):
        """Start the auto-ingestion service"""
        logger.info("🎯 Starting Auto-Ingestion Service")
        
        # Ensure watch directory exists
        Path(self.watch_dir).mkdir(parents=True, exist_ok=True)
        
        # Start file watcher
        self.observer.schedule(
            self.handler,
            self.watch_dir,
            recursive=True
        )
        self.observer.start()
        logger.info(f"👁️ Watching directory: {self.watch_dir}")
        
        # Start background processor
        processor_task = asyncio.create_task(self.processor.start_processing())
        
        try:
            await processor_task
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down auto-ingestion service")
            self.observer.stop()
        
        self.observer.join()
    
    def get_status(self) -> Dict:
        """Get service status"""
        return {
            'service': 'auto-ingestion',
            'status': 'running',
            'watched_directory': self.watch_dir,
            'queue_status': self.queue.get_status(),
            'recent_completed': list(self.queue.completed.values())[-5:],
            'recent_failed': list(self.queue.failed.values())[-5:]
        }

# Standalone service runner
async def run_auto_ingestion_service():
    """Run the auto-ingestion service"""
    service = AutoIngestionService()
    await service.start()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_auto_ingestion_service())
