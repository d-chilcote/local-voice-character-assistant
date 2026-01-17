"""
Modular Voice Assistant Server (v2.1)
=====================================
The central nervous system for the Local AI Voice Assistant.
Enhanced with Skill-based Agentic Architecture.
"""

import io
import sys
import uvicorn
import numpy as np
import soundfile as sf
import subprocess
import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Body
from pydantic import BaseModel
from fastapi.responses import Response, HTMLResponse
from llama_cpp import Llama
from faster_whisper import WhisperModel

from characters import CHARACTERS
from logger_config import setup_logging, get_logger
from config import cfg
from utils import stderr_suppressor
from skills.registry import registry
from core.agent import Agent

# Initialize Logging
setup_logging()
logger = get_logger(__name__)

# --- GLOBAL STATE ---
CURRENT_CHAR = None
CHAT_HISTORY: List[Dict[str, str]] = []

# --- CLI CHARACTER SELECTION ---
def select_character() -> Dict[str, Any]:
    print("\n╔══════════════════════════════════════╗")
    print("║   SELECT YOUR VOICE ASSISTANT        ║")
    print("╠══════════════════════════════════════╣")
    for idx, char in enumerate(CHARACTERS):
        print(f"║ {idx + 1}. {char['name']:<31} ║")
    print("╚══════════════════════════════════════╝")
    
    while True:
        try:
            choice = int(input(f"\nEnter choice (1-{len(CHARACTERS)}): "))
            if 1 <= choice <= len(CHARACTERS):
                return CHARACTERS[choice - 1]
            print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a number.")

if __name__ == "__main__":
    CURRENT_CHAR = select_character()
else:
    CURRENT_CHAR = CHARACTERS[0] 

# --- DERIVED CONFIG ---
MEMORY_FILE = CURRENT_CHAR["memory_file"]
SYSTEM_PROMPT = CURRENT_CHAR["system_prompt"]
HISTORY_LIMIT = cfg().history_limit

# --- VOICE CONFIG ---
def get_voice() -> str:
    target = CURRENT_CHAR["voice_native"]
    if subprocess.run(["say", "-v", target, "test"], capture_output=True).returncode == 0:
        return target
    logger.warning(f"Voice '{target}' not found. Falling back to '{CURRENT_CHAR['voice_fallback']}'.")
    return CURRENT_CHAR["voice_fallback"]

MAC_VOICE = get_voice()

# --- ASCII ART ---
print(CURRENT_CHAR["face"])
logger.info(f"--- LOADING {CURRENT_CHAR['name']} (v2.1 AGENTIC) ---")
logger.info(f"Memorizing to: {MEMORY_FILE}")

app = FastAPI()

# --- LOAD SYSTEMS ---
logger.info("1. Loading Ears (Whisper)...")
whisper = WhisperModel("base.en", device="cpu", compute_type="int8")

logger.info("2. Loading Brain (Llama)...")
with stderr_suppressor():
    llm = Llama(
        model_path=cfg().llama_path, 
        n_gpu_layers=cfg().llama_n_gpu_layers, 
        n_ctx=cfg().llama_n_ctx, 
        n_threads=cfg().llama_n_threads, 
        verbose=False
    )

logger.info(f"3. Voice Engine: {MAC_VOICE}...")

# --- AGENT INITIALIZATION ---
agent = Agent(
    name=CURRENT_CHAR["name"],
    system_prompt=CURRENT_CHAR["system_prompt"],
    llm=llm,
    registry=registry
)
def load_memory() -> None:
    global CHAT_HISTORY
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                CHAT_HISTORY = json.load(f)
            logger.info(f"[MEMORY] Loaded {len(CHAT_HISTORY)} previous thoughts.")
        except Exception as e:
            logger.error(f"[MEMORY] Error loading memory: {e}")
            CHAT_HISTORY = [{"role": "system", "content": SYSTEM_PROMPT}]
    else:
        CHAT_HISTORY = [{"role": "system", "content": SYSTEM_PROMPT}]

def save_memory() -> None:
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(CHAT_HISTORY, f, indent=2)
    except Exception as e:
        logger.error(f"[MEMORY] Error saving memory: {e}")

# --- CLI FLAGS ---
if "--reset" in sys.argv:
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
        logger.info("[MEMORY] Wiped memory due to --reset flag.")

load_memory()
logger.info("--- SYSTEMS READY ---")

@app.get("/", response_class=HTMLResponse)
async def get_harness():
    """Serves the web-based test harness."""
    try:
        with open("harness.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Harness file not found</h1>"

class TextQuery(BaseModel):
    text: str
    character: Optional[str] = None

@app.post("/chat/text")
async def chat_text_endpoint(query: TextQuery):
    """
    Direct text chat endpoint. Useful for testing and CLI/Web harnesses.
    """
    logger.info(f"[TEXT CHAT] Received: {query.text}")
    
    # Process text through the shared agent loop
    final_response = await process_agent_chat(query.text)
    
    return {
        "speech": final_response,
        "history": CHAT_HISTORY[-5:] # Return a slice for UI updates
    }

async def process_agent_chat(user_text: str) -> str:
    """
    Shared logic to handle the agent's reasoning loop for both audio and text.
    """
    global CHAT_HISTORY
    
    # Delegate to Agent
    response_text, CHAT_HISTORY = agent.chat(user_text, CHAT_HISTORY)
    
    # Save History 
    save_memory()
    
    return response_text

@app.post("/chat")
async def chat_endpoint(file: UploadFile = File(...)):
    global CHAT_HISTORY

    # A. RECEIVE AUDIO
    audio_bytes = await file.read()
    with io.BytesIO(audio_bytes) as audio_io:
        data, samplerate = sf.read(audio_io)
        data = data.astype(np.float32)

    # B. TRANSCRIBE
    segments, _ = whisper.transcribe(data, beam_size=1)
    user_text = " ".join([s.text for s in segments]).strip()
    
    if not user_text:
        return Response(content=b"", media_type="audio/wav")

    logger.info(f"User: {user_text}")

    # C. UPDATE HISTORY & DELEGATE TO AGENT
    final_response = await process_agent_chat(user_text)

    # F. SPEAK 
    output_file = "temp_output.wav"
    if os.path.exists(output_file):
        os.remove(output_file)
        
    subprocess.run([
        "say",
        "-v", MAC_VOICE,
        "-o", output_file,
        "--data-format=LEF32@22050",
        final_response
    ])

    with open(output_file, "rb") as f:
        wav_data = f.read()

    return Response(content=wav_data, media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host=cfg().host, port=cfg().port)
