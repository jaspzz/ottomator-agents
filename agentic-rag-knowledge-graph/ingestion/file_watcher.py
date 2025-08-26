import asyncio
import logging
import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import hashlib

# Only import what we know exists
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

class AutoIngestionService:
    """Simplified auto-ingestion service"""
    
    def __init__(self):
        self.queue = IngestionQueue()
        self.topic_ingester = TopicIngestion()
        self.watch_dir = Path("/app/ingestion-watch")
        self.processed_dir = self.watch_dir / "processed"
        self.failed_dir = self.watch_dir / "failed"
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
    
    async def start(self):
        """Start the auto-ingestion service"""
        logger.info("🚀 Starting simplified auto-ingestion processor")
        
        # Just run the processor for now (without file watcher)
        # We'll manually trigger jobs via API
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
                await asyncio.sleep(5)
    
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
            'service': 'simplified-auto-ingestion',
            'status': 'running',
            'watched_directory': str(self.watch_dir),
            'queue_status': self.queue.get_status(),
            'recent_completed': list(self.queue.completed.values())[-5:],
            'recent_failed': list(self.queue.failed.values())[-5:]
        }
