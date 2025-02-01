import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import httpx
from httpx import ReadTimeout

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from supabase import create_client, Client

load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = "deepseek-coder:6.7b"  # Good for technical content
EMBEDDING_MODEL = "nomic-embed-text"

# Rate limiting settings
RATE_LIMIT_PER_MINUTE = 20  # Ollama can handle more requests
RATE_LIMIT_DELAY = 60 / RATE_LIMIT_PER_MINUTE

# Semaphore for rate limiting
rate_limit_semaphore = asyncio.Semaphore(1)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def with_rate_limit(func):
    """Decorator to handle rate limiting"""
    async with rate_limit_semaphore:
        result = await func()
        await asyncio.sleep(RATE_LIMIT_DELAY)
        return result

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using Ollama."""
    async def make_request():
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": CHAT_MODEL,
                        "messages": [
                            {
                                "role": "system",
                                "content": """You are an AI that extracts titles and summaries from documentation chunks.
                                You must respond with ONLY a valid JSON object in this exact format:
                                {"title": "Brief title here", "summary": "Brief summary here"}
                                Do not include any other text or formatting."""
                            },
                            {
                                "role": "user",
                                "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."
                            }
                        ],
                        "stream": False,
                        "format": "json"
                    }
                )
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        content = result['message']['content'].strip()
                        
                        # Debug print
                        print(f"Raw response content: {content}")
                        
                        # Find the JSON object in the response
                        start = content.find('{')
                        end = content.rfind('}') + 1
                        if start >= 0 and end > start:
                            json_str = content[start:end]
                            try:
                                parsed = json.loads(json_str)
                                if 'title' in parsed and 'summary' in parsed:
                                    return parsed
                            except json.JSONDecodeError as e:
                                print(f"JSON parse error: {e}")
                                print(f"Attempted to parse: {json_str}")
                        
                        # If we couldn't parse the JSON, try to extract title and summary directly
                        title_match = content.lower().find('"title"')
                        summary_match = content.lower().find('"summary"')
                        if title_match >= 0 and summary_match >= 0:
                            # Create a simple JSON object
                            return {
                                "title": content.split('"title"')[1].split('"')[2],
                                "summary": content.split('"summary"')[1].split('"')[2]
                            }
                            
                    except Exception as e:
                        print(f"Error parsing response: {e}")
                        print(f"Full response: {result}")
                else:
                    print(f"Error from Ollama API: {response.text}")
                    
        except Exception as e:
            print(f"Error getting title and summary: {e}")
            
        # Return default values if all parsing attempts fail
        return {
            "title": "Error extracting title",
            "summary": "Error extracting summary"
        }

    return await with_rate_limit(make_request)

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Ollama."""
    async def make_request():
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={
                        "model": EMBEDDING_MODEL,
                        "prompt": text
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result['embedding']
                    
                    # Ensure we have a list of floats
                    embedding = [float(x) for x in embedding]
                    
                    # Pad or truncate to 1536 dimensions
                    if len(embedding) > 1536:
                        return embedding[:1536]
                    elif len(embedding) < 1536:
                        embedding.extend([0.0] * (1536 - len(embedding)))
                    return embedding
                else:
                    print(f"Error from Ollama API: {response.text}")
                    return [0.0] * 1536
                    
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0] * 1536

    return await with_rate_limit(make_request)

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    result = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "chunk_size": len(chunk),
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=result["title"],
        summary=result["summary"],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_pydantic_ai_docs_urls() -> List[str]:
    """Get URLs from Pydantic AI docs sitemap."""
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def main():
    # Get URLs from Pydantic AI docs
    urls = get_pydantic_ai_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())