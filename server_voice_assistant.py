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
from utils.network import get_local_ip

# Initialize Logging
setup_logging()
logger = get_logger(__name__)

# --- GLOBAL STATE ---
CURRENT_CHAR = None
CHAT_HISTORY: List[Dict[str, str]] = []
MEMORY_LOCK = asyncio.Lock()

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
    # If starting via CLI, maybe we want to select? 
    # But for a background server, maybe we default.
    # Let's check for a flag or env var, otherwise default to r2_67
    default_char_name = os.getenv("DEFAULT_CHARACTER", "R2-67")
    selected = next((c for c in CHARACTERS if c["name"] == default_char_name), CHARACTERS[0])
    
    if sys.stdin.isatty():
        CURRENT_CHAR = select_character()
    else:
        CURRENT_CHAR = selected
else:
    CURRENT_CHAR = CHARACTERS[0] # R2-67 is first in the list anyway

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

# Add detail logging for 422s
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation Error: {exc.errors()}")
    logger.error(f"Request body: {await request.body()}")
    return Response(content=str(exc.errors()), status_code=422)

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

def _write_memory_to_disk(filename: str, history: List[Dict[str, str]]) -> None:
    try:
        with open(filename, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"[MEMORY] Error saving memory: {e}")

async def save_memory() -> None:
    async with MEMORY_LOCK:
        # Snapshot to ensure thread safety
        history_snapshot = list(CHAT_HISTORY)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write_memory_to_disk, MEMORY_FILE, history_snapshot)

# --- CLI FLAGS ---
if "--reset" in sys.argv:
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
        logger.info("[MEMORY] Wiped memory due to --reset flag.")

load_memory()
logger.info("--- SYSTEMS READY ---")

@app.get("/")
@app.head("/")
async def get_harness():
    """Serves the web-based test harness and handles heartbeats."""
    return Response(content="Voice Assistant Server v2.1", media_type="text/plain")

@app.get("/v1")
@app.head("/v1")
@app.get("/v1/")
@app.head("/v1/")
async def v1_heartbeat():
    """Handles OpenAI discovery heartbeats."""
    return {"status": "ok", "version": "v1"}

@app.get("/api/version")
async def ollama_version():
    """Shim for Ollama version check."""
    return {"version": "0.1.48"}

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

# --- OPENAI COMPATIBILITY ENDPOINTS ---

@app.get("/v1/models")
async def list_models():
    """Returns available characters as models for OpenAI compatibility."""
    return {
        "object": "list",
        "data": [
            {
                "id": char["id"],
                "object": "model",
                "created": 1700000000,
                "owned_by": "local-assistant"
            } for char in CHARACTERS
        ]
    }

class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    stream: Optional[bool] = False

@app.post("/v1/chat/completions")
async def chat_completions(req: OpenAIChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    logger.info(f"[OPENAI API] Request for model: {req.model}")
    
    # Extract last user message
    user_text = ""
    for msg in reversed(req.messages):
        if msg["role"] == "user":
            user_text = msg["content"]
            break
            
    if not user_text:
        return {"error": "No user message found"}

    # Process through agent loop
    # Note: Currently we use the global CHAT_HISTORY. 
    # Enchanted sends the full history, but our agent manages its own memory file.
    # To be truly compatible with external clients that manage history, 
    # we might want to use their history instead of our local one, 
    # but our agent is designed for local persistence.
    # For now, we'll use the user text and let the agent manage context.
    
    final_response = await process_agent_chat(user_text)
    
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_response
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

# --- OLLAMA COMPATIBILITY SHIMS ---

@app.get("/api/tags")
@app.get("/v1/api/tags")
async def ollama_tags():
    """Shim for Ollama model discovery (Enchanted often calls this)."""
    # Ollama Kit (used by Enchanted) is very picky about ISO8601 and details
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return {
        "models": [
            {
                "name": f"{char['id']}:latest",
                "model": f"{char['id']}:latest",
                "modified_at": now,
                "size": 4920734272,
                "digest": f"local-sha256-{char['id']}",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "8.0B",
                    "quantization_level": "Q4_K_M"
                }
            } for char in CHARACTERS
        ]
    }

class OllamaChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False

    class Config:
        extra = "allow" # Be permissive with Enchanted's metadata

@app.post("/api/chat")
@app.post("/v1/api/chat")
async def ollama_chat(req: OllamaChatRequest):
    """Shim for Ollama chat completion."""
    try:
        user_text = req.messages[-1]["content"] if req.messages else ""
        final_response = await process_agent_chat(user_text)
        
        return {
            "model": req.model,
            "created_at": datetime.now().isoformat(),
            "message": {
                "role": "assistant",
                "content": final_response
            },
            "done": True
        }
    except Exception as e:
        logger.error(f"[OLLAMA SHIM ERROR] {e}")
        return Response(content=str(e), status_code=500)

class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: Optional[str] = None
    stream: Optional[bool] = False

    class Config:
        extra = "allow"

@app.post("/api/generate")
@app.post("/v1/api/generate")
async def ollama_generate(req: OllamaGenerateRequest):
    """Shim for Ollama generation."""
    try:
        prompt = req.prompt or ""
        final_response = await process_agent_chat(prompt)
        return {
            "model": req.model,
            "created_at": datetime.now().isoformat(),
            "response": final_response,
            "done": True
        }
    except Exception as e:
        logger.error(f"[OLLAMA SHIM ERROR] {e}")
        return Response(content=str(e), status_code=500)

@app.get("/api/ps")
async def ollama_ps():
    """Shim for running models list."""
    return {"models": []}

@app.post("/api/show")
async def ollama_show(req: Dict[str, Any]):
    """Shim for model details."""
    model_name = req.get("name") or req.get("model", "")
    lookup = model_name.split(":")[0]
    char = next((c for c in CHARACTERS if c["id"] == lookup or c["name"] == lookup), CHARACTERS[0])
    
    return {
        "license": "Local Assistant License",
        "modelfile": f"FROM {lookup}\nSYSTEM \"\"\"{char['system_prompt']}\"\"\"",
        "parameters": "stop \"<|end_of_text|>\"",
        "template": "{{ .System }}\nUSER: {{ .Prompt }}\nASSISTANT: ",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "8.0B",
            "quantization_level": "Q4_K_M"
        },
        "messages": [
            {"role": "system", "content": char["system_prompt"]}
        ]
    }

async def process_agent_chat(user_text: str) -> str:
    """
    Shared logic to handle the agent's reasoning loop for both audio and text.
    """
    global CHAT_HISTORY
    
    # Delegate to Agent
    response_text, CHAT_HISTORY = agent.chat(user_text, CHAT_HISTORY)
    
    # Save History 
    await save_memory()
    
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
    local_ip = get_local_ip()
    port = cfg().port
    logger.info(f"🚀 Server starting on http://{local_ip}:{port}")
    logger.info(f"📱 Connect Enchanted with Base URL: http://{local_ip}:{port}/v1")
    uvicorn.run(app, host=cfg().host, port=port)
