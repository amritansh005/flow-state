from fastapi import FastAPI, HTTPException
from ai_talker_functions_multi import ConversationStep, step_prompts, available_steps, run_conversation_step, save_to_json
from typing import List
from pydantic import BaseModel
import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Create a directory to store conversation data
os.makedirs("conversation_data", exist_ok=True)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Multi-Question Conversation Flow API"}

class ConversationRequest(BaseModel):
    steps: List[str]

@app.get("/available_steps")
async def get_available_steps():
    return {"steps": list(available_steps.keys())}

@app.post("/conversation/")
async def handle_conversation(request: ConversationRequest):
    """API endpoint to handle multi-question conversation flow dynamically."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conversation_sequence = [available_steps[step] for step in request.steps]
    conversation = []

    for step in conversation_sequence:
        print(f"\nStarting {step.function}")
        conversation = await run_conversation_step(step, conversation)

    # Save the entire conversation
    conversation_filename = f"conversation_data/{timestamp}_conversation_multi.json"
    save_to_json(conversation, conversation_filename)
    print(f"\nEntire conversation saved to {conversation_filename}")

    return {"response": conversation, "saved_file": conversation_filename}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
