from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import os
import nest_asyncio
import uvicorn
from datetime import datetime
from pymongo import MongoClient
from utils import mongo_module

# MongoDB Setup
URI = mongo_module.mongo_creds_from_config()
client = mongo_module.create_mongo_client(URI)

# Initialize FastAPI
app = FastAPI()

# Pydantic model for input validation
class ChatRequest(BaseModel):
    user_id: str
    prompt: str

@app.post("/generate")
def generate(request: ChatRequest):
    user_id = request.user_id
    prompt = request.prompt
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

    # Call the LLM model
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    client_ollama = ollama.Client(host=ollama_host)
    response = client_ollama.chat(model="llama3.2:1b", messages=messages)
    model_response = response['message']['content']

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
        "timestamp": timestamp  # Explicitly return the timestamp here
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

# Apply nest_asyncio for running in Jupyter (only needed if using Jupyter)
nest_asyncio.apply()

# Start FastAPI server (use only when running script directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
