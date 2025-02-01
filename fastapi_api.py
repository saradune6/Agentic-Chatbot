from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
import numpy as np
import httpx
import json
import asyncio
from datetime import datetime

load_dotenv()

app = FastAPI(title="PydanticAI Documentation Search API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize clients
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "deepseek-r1"

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class SearchQuery(BaseModel):
    query: str
    num_results: Optional[int] = 5

class SearchResult(BaseModel):
    url: str
    title: str
    summary: str
    content: str
    similarity: float

async def chat_with_ollama(prompt: str) -> httpx.Response:
    """Send a chat request to Ollama API."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": "I am an AI assistant specialized in helping with Pydantic and Python development."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": True
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.text}")
            
        return response

async def process_ollama_stream(response: httpx.Response):
    """Process streaming response from Ollama."""
    async for line in response.aiter_lines():
        if line:
            try:
                data = json.loads(line)
                if 'message' in data and 'content' in data['message']:
                    content = data['message']['content']
                    yield json.dumps({
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    }) + "\n"
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                continue
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue

@app.get("/")
async def root():
    return {"message": "Welcome to PydanticAI Documentation Search API"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await chat_with_ollama(request.message)
        return StreamingResponse(
            process_ollama_stream(response),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[SearchResult])
async def search(query: SearchQuery):
    try:
        # Get embedding for the search query using Ollama
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "prompt": query.query
                }
            )
            
            if response.status_code != 200:
                raise Exception("Failed to get embedding from Ollama")
                
            query_embedding = response.json()['embedding']

        # Perform vector similarity search
        rpc_response = supabase.rpc(
            'match_page_sections',
            {
                'query_embedding': query_embedding,
                'match_count': query.num_results
            }
        ).execute()

        results = []
        for item in rpc_response.data:
            similarity = round(float(item['similarity']), 3)
            result = SearchResult(
                url=item['url'],
                title=item['title'],
                summary=item['summary'],
                content=item['content'],
                similarity=similarity
            )
            results.append(result)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    try:
        # Get total number of documents
        count_response = supabase.table("site_pages").select("*", count="exact").execute()
        total_documents = count_response.count if count_response.count is not None else 0

        # Get unique URLs
        urls_response = supabase.table("site_pages").select("url").execute()
        unique_urls = len(set(item['url'] for item in urls_response.data))

        # Get average chunk size with better error handling
        chunks_response = supabase.table("site_pages").select("metadata").execute()
        chunk_sizes = []
        for item in chunks_response.data:
            try:
                if isinstance(item['metadata'], str):
                    metadata = json.loads(item['metadata'])
                else:
                    metadata = item['metadata']
                
                if isinstance(metadata, dict) and 'chunk_size' in metadata:
                    chunk_sizes.append(metadata['chunk_size'])
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Error parsing chunk: {e}")
                continue

        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0

        return {
            "total_documents": total_documents,
            "unique_urls": unique_urls,
            "average_chunk_size": round(avg_chunk_size, 2),
            "processed_chunks": len(chunk_sizes)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        # Check Supabase connection
        supabase.table("site_pages").select("id").limit(1).execute()
        # Check Ollama connection
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/version")
            if response.status_code != 200:
                raise Exception("Ollama not responding")
        
        return {
            "status": "healthy",
            "database": "connected",
            "llm_service": "connected",
            "timestamp": str(np.datetime64('now'))
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 