import asyncio
import logging
import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import hashlib

# Add file watcher imports
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .topic_ingest import TopicIngestion

logger = logging.getLogger(__name__)

class AutoIngestionHandler(FileSystemEventHandler):
    """Handle file system events for auto-ingestion"""
    
    def __init__(self, queue):
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
    
    def on_moved(self, event):
        """Handle file moves (SFTP often moves files)"""
        if not event.is_directory:
            asyncio.create_task(self.process_new_file(event.dest_path))
    
    async def process_new_file(self, file_path: str):
        """Process newly detected file"""
        try:
            path = Path(file_path)
            
            # Skip hidden files, temp files, and unsupported formats
            if (path.name.startswith('.') or 
                path.name.startswith('~') or 
                path.suffix not in self.supported_extensions):
                logger.info(f"⏭️ Skipping unsupported file: {path.name}")
                return
            
            # Determine topic from folder structure
            try:
                relative_path = path.relative_to(self.watch_dir)
                topic_folder = relative_path.parts[0] if relative_path.parts else 'general'
            except ValueError:
                # File is not in watch directory
                logger.warning(f"⚠️ File not in watch directory: {path}")
                return
                
            topic = self.topic_mapping.get(topic_folder, 'general')
            
            # Skip if file is in processed/failed folders
            if topic_folder in ['processed', 'failed']:
                return
            
            # Check if file is stable (not still being uploaded)
            if not await self.is_file_stable(file_path):
                logger.info(f"⏳ File still uploading: {path.name}")
                # Schedule retry
                await asyncio.sleep(5)
                await self.process_new_file(file_path)
                return
            
            # Add to ingestion queue
            job_id = await self.queue.add_job(str(path), topic)
            logger.info(f"📁 New file detected: {path.name} → Topic: {topic} → Job: {job_id}")
            
        except Exception as e:
            logger.error(f"❌ Error processing new file {file_path}: {e}")
    
    async def is_file_stable(self, file_path: str, wait_time: int = 2) -> bool:
        """Check if file size is stable (upload complete)"""
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            
            size1 = path.stat().st_size
            await asyncio.sleep(wait_time)
            
            if not path.exists():  # File might have been moved
                return False
                
            size2 = path.stat().st_size
            
            # File is stable if size hasn't changed and is > 0
            is_stable = size1 == size2 and size1 > 0
            logger.info(f"📊 File stability check for {path.name}: {size1} → {size2} bytes, stable: {is_stable}")
            return is_stable
            
        except Exception as e:
            logger.error(f"❌ Error checking file stability: {e}")
            return False

# Update the AutoIngestionService class
class AutoIngestionService:
    """Auto-ingestion service with file watcher"""
    
    def __init__(self):
        self.queue = IngestionQueue()
        self.topic_ingester = TopicIngestion()
        self.watch_dir = Path("/app/ingestion-watch")
        self.processed_dir = self.watch_dir / "processed"
        self.failed_dir = self.watch_dir / "failed"
        
        # File watcher components
        self.handler = AutoIngestionHandler(self.queue)
        self.observer = Observer()
        
        # Create directories
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
    
    async def start(self):
        """Start the auto-ingestion service with file watcher"""
        logger.info("🚀 Starting auto-ingestion service with file watcher")
        
        # Start file watcher
        try:
            self.observer.schedule(
                self.handler,
                str(self.watch_dir),
                recursive=True
            )
            self.observer.start()
            logger.info(f"👁️ File watcher started, monitoring: {self.watch_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to start file watcher: {e}")
        
        # Process any existing files first
        await self.scan_existing_files()
        
        # Start background processor
        logger.info("⚙️ Starting background processor")
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
    
    async def scan_existing_files(self):
        """Scan for existing files in watch directories"""
        logger.info("🔍 Scanning for existing files...")
        
        topic_folders = ['business-central', 'ai-research', 'cloud-computing', 'general']
        found_files = 0
        
        for topic in topic_folders:
            topic_dir = self.watch_dir / topic
            if topic_dir.exists():
                for file_path in topic_dir.glob("*"):
                    if file_path.is_file() and file_path.suffix in ['.md', '.txt', '.pdf', '.docx']:
                        job_id = await self.queue.add_job(str(file_path), topic, priority=1)
                        found_files += 1
                        logger.info(f"📋 Queued existing file: {file_path.name} → {topic} → Job: {job_id}")
        
        if found_files > 0:
            logger.info(f"✅ Found and queued {found_files} existing files")
        else:
            logger.info("📭 No existing files found")
    
    # ... (rest of the methods remain the same as before)
