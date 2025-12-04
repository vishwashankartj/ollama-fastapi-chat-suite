from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import os
import nest_asyncio
import uvicorn
from datetime import datetime
from pymongo import MongoClient
from utils import mongo_module
import google.generativeai as genai
from typing import Optional

# MongoDB Setup
URI = mongo_module.mongo_creds_from_config()
client = mongo_module.create_mongo_client(URI)

# Initialize FastAPI with Swagger at root
app = FastAPI(
    title="Ollama LLM Chat API",
    description="API for chatting with LLMs using Ollama or Gemini",
    version="1.0.0",
    docs_url="/",
    redoc_url="/redoc"
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Pydantic model for input validation
class ChatRequest(BaseModel):
    user_id: str
    prompt: str
    provider: Optional[str] = "ollama"  # "ollama" or "gemini"
    model: Optional[str] = None  # Optional model override

def call_ollama(messages: list, model: str = "llama3.2:1b") -> str:
    """Call Ollama API"""
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    client_ollama = ollama.Client(host=ollama_host)
    response = client_ollama.chat(model=model, messages=messages)
    return response['message']['content']

def call_gemini(messages: list, model: str = "gemini-1.5-flash") -> str:
    """Call Gemini API"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    # Convert messages to Gemini format
    gemini_model = genai.GenerativeModel(model)
    
    # Build conversation history for Gemini
    chat_parts = []
    for msg in messages:
        if msg["role"] == "system":
            # Gemini doesn't have system role, prepend to first user message
            continue
        role = "user" if msg["role"] == "user" else "model"
        chat_parts.append({"role": role, "parts": [msg["content"]]})
    
    # Start chat with history
    chat = gemini_model.start_chat(history=chat_parts[:-1] if len(chat_parts) > 1 else [])
    
    # Send the latest message
    response = chat.send_message(chat_parts[-1]["parts"][0] if chat_parts else messages[-1]["content"])
    return response.text

@app.post("/generate")
def generate(request: ChatRequest):
    user_id = request.user_id
    prompt = request.prompt
    provider = request.provider
    timestamp = datetime.utcnow()

    # Prepare the user message with timestamp
    user_message = {"role": "user", "content": prompt, "timestamp": timestamp}

    # Fetch previous chat history from MongoDB
    chat_history = list(mongo_module.get_conversation(client, "personalAI", "chatHistory", user_id))

    # Append the new message
    chat_history.append(user_message)

    # Prepare system message
    system_prompt = {"role": "system", "content": "You are a patient and knowledgeable teacher."}

    # Build message list for LLM
    messages = [system_prompt] + chat_history

    # Call the appropriate LLM provider
    try:
        if provider == "gemini":
            model = request.model or "gemini-1.5-flash"
            model_response = call_gemini(messages, model)
        else:  # Default to ollama
            model = request.model or "llama3.2:1b"
            model_response = call_ollama(messages, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling {provider}: {str(e)}")

    # Store assistant response with timestamp
    assistant_message = {"role": "assistant", "content": model_response, "timestamp": datetime.utcnow()}
    chat_history.append(assistant_message)

    # Store updated chat history in MongoDB
    mongo_module.insert_data(client, "personalAI", "chatHistory", {
        "user_id": user_id,
        "conversation": chat_history
    })

    # Return response including model response, chat history, and timestamp
    return {
        "response": model_response,
        "chat_history": chat_history,
        "timestamp": timestamp,
        "provider": provider
    }


@app.get("/history/{user_id}")
def get_chat_history(user_id: str):
    chat_history = mongo_module.get_conversation(client, "personalAI", "chatHistory", user_id)
    # Only keep the last 10 conversations
    chat_history = chat_history[-10:]
    return {"history": chat_history}


@app.post("/reset")
def reset_session(user_id: str):
    result = mongo_module.delete_data(client, "personalAI", "chatHistory", {"user_id": user_id})
    
    if result.deleted_count > 0:
        return {"message": f"Session for {user_id} has been reset."}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


# OpenAI-compatible endpoints for Open WebUI integration
class OpenAIChatMessage(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class OpenAIChatChoice(BaseModel):
    index: int
    message: OpenAIChatMessage
    finish_reason: str

class OpenAIChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIChatChoice]

class OpenAIModel(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str

class OpenAIModelsResponse(BaseModel):
    object: str
    data: list[OpenAIModel]

@app.get("/v1/models")
def list_models():
    """List available Gemini models in OpenAI format"""
    models = [
        {
            "id": "gemini-1.5-flash",
            "object": "model",
            "created": 1234567890,
            "owned_by": "google"
        },
        {
            "id": "gemini-1.5-pro",
            "object": "model",
            "created": 1234567890,
            "owned_by": "google"
        },
        {
            "id": "gemini-2.0-flash-exp",
            "object": "model",
            "created": 1234567890,
            "owned_by": "google"
        }
    ]
    return {"object": "list", "data": models}

@app.post("/v1/chat/completions")
def openai_chat_completions(request: OpenAIChatRequest):
    """OpenAI-compatible chat completions endpoint for Gemini"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    # Convert OpenAI messages to our internal format
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # Call Gemini
    try:
        response_text = call_gemini(messages, request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Gemini: {str(e)}")
    
    # Return OpenAI-compatible response
    return {
        "id": f"chatcmpl-{datetime.utcnow().timestamp()}",
        "object": "chat.completion",
        "created": int(datetime.utcnow().timestamp()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ]
    }


# Apply nest_asyncio for running in Jupyter (only needed if using Jupyter)
nest_asyncio.apply()

# Start FastAPI server (use only when running script directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
