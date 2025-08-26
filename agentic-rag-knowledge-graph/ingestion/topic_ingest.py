import asyncio
import asyncpg
import os
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class TopicIngestion:
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL')
        
    async def create_topic(self, name: str, description: str, slug: str) -> str:
        """Create a new topic and return its ID"""
        conn = await asyncpg.connect(self.db_url)
        try:
            result = await conn.fetchrow("""
                INSERT INTO topics (name, description, slug) 
                VALUES ($1, $2, $3) 
                ON CONFLICT (slug) DO UPDATE SET 
                    name = EXCLUDED.name,
                    description = EXCLUDED.description
                RETURNING id
            """, name, description, slug)
            return result['id']
        finally:
            await conn.close()
    
    async def get_topic_id(self, slug: str) -> Optional[str]:
        """Get topic ID by slug"""
        conn = await asyncpg.connect(self.db_url)
        try:
            result = await conn.fetchrow(
                "SELECT id FROM topics WHERE slug = $1", slug
            )
            return result['id'] if result else None
        finally:
            await conn.close()
    
    async def ingest_topic_documents(
        self, 
        topic_slug: str,
        documents_path: str,
        clean: bool = False
    ):
        """Ingest documents for a specific topic"""
        
        # Get or create topic
        topic_id = await self.get_topic_id(topic_slug)
        if not topic_id:
            raise ValueError(f"Topic '{topic_slug}' not found. Create it first.")
        
        logger.info(f"📚 Ingesting documents for topic: {topic_slug}")
        
        # Clean existing topic data if requested
        if clean:
            await self.clean_topic_data(topic_id)
        
        # Process documents in the specified path
        docs_path = Path(documents_path)
        if not docs_path.exists():
            raise ValueError(f"Documents path not found: {documents_path}")
        
        processed_count = 0
        
        # Process all supported files in the directory
        for doc_path in docs_path.glob("*"):
            if doc_path.is_file() and doc_path.suffix in ['.md', '.txt']:
                try:
                    logger.info(f"📄 Processing: {doc_path.name}")
                    await self.process_document_with_topic(doc_path, topic_id)
                    processed_count += 1
                except Exception as e:
                    logger.error(f"❌ Failed to process {doc_path.name}: {e}")
        
        logger.info(f"✅ Successfully processed {processed_count} documents for topic '{topic_slug}'")
    
    async def process_document_with_topic(self, doc_path: Path, topic_id: str):
        """Process a single document with topic assignment"""
        conn = await asyncpg.connect(self.db_url)
        try:
            # Read document
            content = doc_path.read_text(encoding='utf-8')
            
            # Insert document with topic
            doc_id = await conn.fetchval("""
                INSERT INTO documents (title, source, content, topic_id, metadata)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """, 
                doc_path.stem,  # title
                str(doc_path),  # source
                content,        # content
                topic_id,       # topic_id
                {'file_type': doc_path.suffix[1:], 'original_path': str(doc_path)}  # metadata
            )
            
            # Simple chunking - split by double newlines and paragraphs
            chunks = self.simple_chunk_text(content)
            
            # Process each chunk
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) < 50:  # Skip very short chunks
                    continue
                    
                # For now, skip embedding generation since it depends on external functions
                # You can add this later when the embedding service is properly set up
                
                # Insert chunk with topic (without embedding for now)
                await conn.execute("""
                    INSERT INTO chunks (document_id, content, chunk_index, topic_id, metadata, token_count)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    doc_id,
                    chunk_text,
                    i,
                    topic_id,
                    {'chunk_method': 'simple'},
                    len(chunk_text.split())
                )
            
            logger.info(f"✅ Processed {doc_path.name}: {len(chunks)} chunks")
            
        finally:
            await conn.close()
    
    def simple_chunk_text(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Simple text chunking by paragraphs and size"""
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, save current chunk
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def clean_topic_data(self, topic_id: str):
        """Clean all data for a specific topic"""
        conn = await asyncpg.connect(self.db_url)
        try:
            # Delete chunks first (foreign key constraint)
            deleted_chunks = await conn.fetchval("DELETE FROM chunks WHERE topic_id = $1 RETURNING count(*)", topic_id)
            # Delete documents
            deleted_docs = await conn.fetchval("DELETE FROM documents WHERE topic_id = $1 RETURNING count(*)", topic_id)
            logger.info(f"🧹 Cleaned topic data: {deleted_docs} documents, {deleted_chunks} chunks")
        finally:
            await conn.close()
    
    async def list_topics(self):
        """List all available topics"""
        conn = await asyncpg.connect(self.db_url)
        try:
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
            
            print("\n📚 Available Topics:")
            print("=" * 60)
            for topic in topics:
                print(f"📁 Name: {topic['name']}")
                print(f"🔗 Slug: {topic['slug']}")
                print(f"📝 Description: {topic['description']}")
                print(f"📄 Documents: {topic['document_count']}")
                print(f"🔤 Chunks: {topic['chunk_count']}")
                print("-" * 40)
            
            return topics
            
        finally:
            await conn.close()

# CLI interface
async def main():
    parser = argparse.ArgumentParser(description='Topic-based document ingestion')
    parser.add_argument('command', choices=['create-topic', 'ingest', 'list-topics', 'clean-topic'])
    parser.add_argument('--topic', help='Topic slug')
    parser.add_argument('--name', help='Topic name (for create-topic)')
    parser.add_argument('--description', help='Topic description (for create-topic)')
    parser.add_argument('--documents', help='Path to documents directory')
    parser.add_argument('--clean', action='store_true', help='Clean existing data before ingestion')
    
    args = parser.parse_args()
    
    ingester = TopicIngestion()
    
    if args.command == 'create-topic':
        if not all([args.topic, args.name]):
            print("❌ --topic and --name are required for create-topic")
            return
        
        topic_id = await ingester.create_topic(
            args.name, 
            args.description or "", 
            args.topic
        )
        print(f"✅ Created topic: {args.name} (ID: {topic_id})")
    
    elif args.command == 'ingest':
        if not all([args.topic, args.documents]):
            print("❌ --topic and --documents are required for ingest")
            return
        
        await ingester.ingest_topic_documents(
            args.topic, 
            args.documents, 
            args.clean
        )
    
    elif args.command == 'list-topics':
        await ingester.list_topics()
    
    elif args.command == 'clean-topic':
        if not args.topic:
            print("❌ --topic is required for clean-topic")
            return
        
        topic_id = await ingester.get_topic_id(args.topic)
        if topic_id:
            await ingester.clean_topic_data(topic_id)
            print(f"✅ Cleaned data for topic: {args.topic}")
        else:
            print(f"❌ Topic not found: {args.topic}")

if __name__ == "__main__":
    asyncio.run(main())
