from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import httpx

import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
# logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          


# Initialize Ollama model
ollama_model = OllamaModel(
    model_name="deepseek-r1",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)

# Initialize the AI expert agent with Ollama
pydantic_ai_expert = Agent(
    name="Pydantic AI Expert",
    system_prompt="I am an AI assistant specialized in helping with Pydantic and Python development.",
    model=ollama_model
)

# Initialize Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "deepseek-r1"

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

async def run_agent_with_streaming(user_input: str):
    # Create placeholder for streaming output
    message_placeholder = st.empty()
    full_response = []
    
    try:
        response = await chat_with_ollama(user_input)
        async for line in response.aiter_lines():
            if line:
                try:
                    # Parse the streaming response
                    data = json.loads(line)
                    if 'message' in data and 'content' in data['message']:
                        content = data['message']['content']
                        full_response.append(content)
                        # Join all response parts and display
                        message_placeholder.markdown(''.join(full_response))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    continue
    
    except Exception as e:
        st.error(f"Error communicating with Ollama: {e}")
        return "I apologize, but I encountered an error while processing your request."
    
    final_response = ''.join(full_response)
    return final_response if final_response else "No response generated"

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

async def main():
    st.title("Pydantic AI Expert")
    initialize_session_state()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    user_input = st.chat_input("What would you like to know about Pydantic?")

    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display assistant response with streaming
        with st.chat_message("assistant"):
            response = await run_agent_with_streaming(user_input)
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})

def run():
    asyncio.run(main())

def get_stats():
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
                # Handle both string and dict metadata
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
            "Total Documents": total_documents,
            "Unique URLs": unique_urls,
            "Average Chunk Size": f"{round(avg_chunk_size, 2)} characters",
            "Processed Chunks": len(chunk_sizes)
        }

    except Exception as e:
        print(f"Error getting stats: {e}")
        return {
            "Error": f"Failed to get stats: {str(e)}"
        }

if __name__ == "__main__":
    run()
