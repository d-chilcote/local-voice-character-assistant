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
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Body
from pydantic import BaseModel, ConfigDict
from fastapi.responses import Response, HTMLResponse
from llama_cpp import Llama
from faster_whisper import WhisperModel

from characters import CHARACTERS
from logger_config import setup_logging, get_logger
from config import cfg
from utils import stderr_suppressor
from skills.registry import registry
from langgraph.checkpoint.memory import MemorySaver
from core.agent_graph import create_agent_graph, AgentState
from core.llm_langchain import create_chat_llm
from core.tools_bridge import get_all_tools
from core.llm import BaseLLM, LlamaCPPLLM
from utils.network import get_local_ip
from langchain_core.messages import HumanMessage

# Initialize Logging
setup_logging()
logger = get_logger(__name__)

# --- GLOBAL STATE (INITIALIZED AT RUNTIME) ---
if "pytest" in sys.modules:
    from unittest.mock import MagicMock
    whisper = MagicMock()
else:
    whisper = None

llm = None
langchain_llm = None
agent_graph = None
MAC_VOICE = "Alex" # Default safe voice for tests
MEMORY_FILE = "mem.json"
SYSTEM_PROMPT = "You are a helpful assistant."
HISTORY_LIMIT = 10

CURRENT_CHAR = CHARACTERS[0] # Default to first character
CHAT_HISTORY: List[Dict[str, str]] = []  # Legacy, kept for compatibility
MEMORY_LOCK = asyncio.Lock()
SELECTED_LLM: Optional[BaseLLM] = None
SELECTED_MODEL_PATH: Optional[str] = None  # Track path for LangChain
SELECTED_MODEL_CONFIG: Dict[str, Any] = {}  # Model-specific config for LangChain
memory_saver = MemorySaver() # LangGraph Checkpointer

# --- CLI MODEL SELECTION ---
def select_model() -> tuple[BaseLLM, str, Dict[str, Any]]:
    """Prompts user to select a local GGUF model. Returns (llm, path, config_dict)."""
    options = []
    
    # 1. Local GGUF Files (search ~/ai/models/ only)
    search_path = os.path.expanduser("~/ai/models/")
    if os.path.exists(search_path):
        try:
            gguf_files = [f for f in os.listdir(search_path) if f.endswith(".gguf")]
            for f in gguf_files:
                full_path = os.path.join(search_path, f)
                options.append({
                    "name": f"Local: {f} (via LlamaCPP)",
                    "type": "llama-cpp",
                    "path": full_path
                })
        except Exception as e:
            logger.warning(f"Error scanning for GGUF files in {search_path}: {e}")
    
    if not options:
        logger.error("No GGUF models found.")
        sys.exit(1)

    print("\n╔══════════════════════════════════════╗")
    print("║   SELECT BRAIN (LLM BACKEND)         ║")
    print("╠══════════════════════════════════════╣")
    for idx, opt in enumerate(options):
        print(f"║ {idx + 1}. {opt['name']:<31} ║")
    print("╚══════════════════════════════════════╝")

    while True:
        try:
            choice = int(input(f"\nEnter choice (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                selected = options[choice - 1]
                if selected["type"] == "llama-cpp":
                    # Check for model-specific flags
                    model_path_lower = selected["path"].lower()
                    is_nemotron = "nemotron" in model_path_lower
                    is_devstral = "devstral" in model_path_lower
                    is_qwen = "qwen" in model_path_lower
                    
                    # Global "Mac Studio" Optimized Defaults
                    n_ctx = cfg().llama_n_ctx
                    n_gpu_layers = 99
                    n_threads = 8
                    n_batch = 512 # Default batch size
                    flash_attn = True
                    use_mlock = True
                    chat_format = None
                    default_inference_params = {}

                    if is_nemotron:
                        # Nemotron is large MoE model - use smaller context to avoid memory issues
                        n_ctx = 32768
                        chat_format = "chatml-function-calling"
                        default_inference_params = {
                            "temperature": 0.3,
                            "repeat_penalty": 1.1
                        }
                        logger.info(f"Applying Nemotron optimization flags for {selected['path']} (n_ctx={n_ctx})")
                    
                    elif is_devstral:
                        n_ctx = 131072
                        chat_format = "mistral-instruct"  # Devstral uses Mistral chat format
                        default_inference_params = {
                            "temperature": 0.1,
                            "min_p": 0.1,
                            "repeat_penalty": 1.05
                        }
                        logger.info(f"Applying Devstral optimization flags for {selected['path']}")
                    
                    elif is_qwen:
                        # Check for Qwen 2.5 14B specifically for user-requested optimizations
                        is_qwen_25_14b = "qwen2.5-14b" in model_path_lower
                        
                        if is_qwen_25_14b:
                            n_ctx = 32768
                            n_threads = 10
                            n_batch = 512
                            chat_format = "chatml-function-calling"
                            default_inference_params = {
                                "temperature": 0.6,
                                "min_p": 0.1
                            }
                            logger.info(f"Applying Qwen2.5-14B specific optimization flags for {selected['path']}")
                        else:
                            n_ctx = 131072
                            chat_format = "chatml-function-calling"
                            default_inference_params = {
                                "temperature": 0.6,
                                "min_p": 0.1
                            }
                            logger.info(f"Applying default Qwen optimization flags for {selected['path']}")

                    try:
                        llm_instance = LlamaCPPLLM(
                            model_path=selected["path"],
                            n_ctx=n_ctx,
                            n_gpu_layers=n_gpu_layers,
                            n_threads=n_threads,
                            flash_attn=flash_attn,
                            use_mlock=use_mlock,
                            chat_format=chat_format,
                            extra_config={"n_batch": n_batch},
                            default_inference_params=default_inference_params
                        )
                        # Prepare config for LangChain initialization
                        model_config = {
                            "chat_format": chat_format,
                            "n_threads": n_threads,
                            "n_batch": n_batch,
                            "n_ctx": n_ctx,
                            "temperature": default_inference_params.get("temperature", 0.7),
                            "min_p": default_inference_params.get("min_p", 0.1)
                        }
                        return llm_instance, selected["path"], model_config
                    except Exception as model_error:
                        logger.error(f"Failed to load model: {model_error}")
                        print(f"Error loading model: {model_error}")
                        continue
            print("Invalid number.")
        except ValueError:
            print("Enter a number.")

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown of heavy models and resources.
    This prevents Metal backend crashes and memory leaks during test collection.
    """
    global whisper, llm, langchain_llm, agent_graph, MAC_VOICE, MEMORY_FILE, SYSTEM_PROMPT, HISTORY_LIMIT, CURRENT_CHAR, SELECTED_LLM, SELECTED_MODEL_PATH, SELECTED_MODEL_CONFIG

    # 1. Select Character & Model (if CLI or default)
    default_char_name = os.getenv("DEFAULT_CHARACTER", "R2-67")
    selected_char = next((c for c in CHARACTERS if c["name"] == default_char_name), CHARACTERS[0])
    
    if sys.stdin.isatty():
        SELECTED_LLM, SELECTED_MODEL_PATH, SELECTED_MODEL_CONFIG = select_model()
        CURRENT_CHAR = select_character()
    else:
        SELECTED_MODEL_PATH = cfg().llama_path
        SELECTED_MODEL_CONFIG = {
            "n_ctx": cfg().llama_n_ctx,
            "n_gpu_layers": cfg().llama_n_gpu_layers,
            "flash_attn": True,
            "use_mlock": True,
            "temperature": 0.7,
        }
        SELECTED_LLM = LlamaCPPLLM(
            model_path=SELECTED_MODEL_PATH, 
            n_gpu_layers=cfg().llama_n_gpu_layers, 
            n_ctx=cfg().llama_n_ctx, 
            n_threads=cfg().llama_n_threads
        )
        CURRENT_CHAR = selected_char

    # 2. Assign Derived Config
    MEMORY_FILE = CURRENT_CHAR["memory_file"]
    SYSTEM_PROMPT = CURRENT_CHAR["system_prompt"]
    HISTORY_LIMIT = cfg().history_limit

    # 3. Voice Engine
    target_voice = CURRENT_CHAR["voice_native"]
    if subprocess.run(["say", "-v", target_voice, "test"], capture_output=True).returncode == 0:
        MAC_VOICE = target_voice
    else:
        logger.warning(f"Voice '{target_voice}' not found. Falling back to '{CURRENT_CHAR['voice_fallback']}'.")
        MAC_VOICE = CURRENT_CHAR["voice_fallback"]

    # 4. ASCII Art & Startup Greeting
    print(CURRENT_CHAR["face"])
    logger.info(f"--- LOADING {CURRENT_CHAR['name']} (v2.1 AGENTIC) ---")
    logger.info(f"Memorizing to: {MEMORY_FILE}")

    # 5. Load Ears (Whisper)
    logger.info("1. Loading Ears (Whisper)...")
    whisper = WhisperModel("base.en", device="cpu", compute_type="int8")

    # 6. Load Brain (LLM & LangChain)
    logger.info("2. Ensuring Brain is Ready...")
    llm = SELECTED_LLM
    
    logger.info("3. Initializing LangGraph Agent...")
    langchain_llm = create_chat_llm(
        model_path=SELECTED_MODEL_PATH or cfg().llama_path,
        n_ctx=SELECTED_MODEL_CONFIG.get("n_ctx", cfg().llama_n_ctx),
        n_gpu_layers=SELECTED_MODEL_CONFIG.get("n_gpu_layers", cfg().llama_n_gpu_layers),
        n_threads=SELECTED_MODEL_CONFIG.get("n_threads", cfg().llama_n_threads),
        n_batch=SELECTED_MODEL_CONFIG.get("n_batch", 512),
        temperature=SELECTED_MODEL_CONFIG.get("temperature", 0.7),
        flash_attn=SELECTED_MODEL_CONFIG.get("flash_attn", True),
        use_mlock=SELECTED_MODEL_CONFIG.get("use_mlock", True),
        chat_format=SELECTED_MODEL_CONFIG.get("chat_format", "chatml-function-calling"),
    )
    tools = get_all_tools(history=CHAT_HISTORY, memory_file=MEMORY_FILE, memory_saver=memory_saver)
    agent_graph = create_agent_graph(
        llm=langchain_llm,
        tools=tools,
        system_prompt=CURRENT_CHAR["system_prompt"],
        checkpointer=memory_saver
    )
    logger.info(f"Agent initialized with {len(tools)} tools")

    # 7. Load Memory
    if "--reset" in sys.argv:
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
            logger.info("[MEMORY] Wiped memory due to --reset flag.")
    
    load_memory()
    logger.info("--- SYSTEMS READY ---")

    yield
    
    # Optional Shutdown Logic
    logger.info("Shutting down assistant...")

app = FastAPI(lifespan=lifespan)

# Add detail logging for 422s
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation Error: {exc.errors()}")
    logger.error(f"Request body: {await request.body()}")
    return Response(content=str(exc.errors()), status_code=422)

# System components are initialized within lifespan now
def load_memory() -> None:
    global CHAT_HISTORY
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
                CHAT_HISTORY[:] = data # Update in-place to preserve references
            logger.info(f"[MEMORY] Loaded {len(CHAT_HISTORY)} previous thoughts.")
        except Exception as e:
            logger.error(f"[MEMORY] Error loading memory: {e}")
            CHAT_HISTORY[:] = [{"role": "system", "content": SYSTEM_PROMPT}]
    else:
        CHAT_HISTORY[:] = [{"role": "system", "content": SYSTEM_PROMPT}]

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

# Load memory is now called inside lifespan

@app.get("/")
@app.head("/")
async def get_harness():
    """Serves the web-based test harness and handles heartbeats."""
    return Response(content="Voice Assistant Server v2.1", media_type="text/plain")

# --- WEB CHAT HARNESS ---
from pathlib import Path
HARNESS_PATH = Path(__file__).parent / "harness.html"

@app.get("/chat")
async def get_chat_harness():
    """Serves the web-based chat harness for mobile/desktop browsers."""
    if not HARNESS_PATH.exists():
        return Response(content="Chat harness not found", status_code=404)
    return Response(content=HARNESS_PATH.read_text(), media_type="text/html")

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

    model_config = ConfigDict(extra="allow") # Be permissive with Enchanted's metadata

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

    model_config = ConfigDict(extra="allow")

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

def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

async def process_agent_chat(user_text: str) -> str:
    """
    Shared logic to handle the agent's reasoning loop using LangGraph.
    """
    import re
    
    def clean_response(text: str) -> str:
        """Strip model artifacts from response text."""
        if not text:
            return text
        # Remove <tool_call>...</tool_call> tags and content
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
        # Remove <think>...</think> tags and content
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove any remaining unclosed tags
        text = re.sub(r'</?tool_call>', '', text)
        text = re.sub(r'</?think>', '', text)
        # Remove Nemotron <function> tags and content
        text = re.sub(r'<function=\w+>.*?</function>', '', text, flags=re.DOTALL)
        # Remove Nemotron's chain-of-thought thinking prefixes
        # These patterns indicate internal reasoning, not user-facing response
        thinking_patterns = [
            r'^We need to respond.*?\.(?:\s|$)',
            r'^We have the weather.*?(?=(?:[A-Z]|$))',
            r'^We can (?:use|answer|call).*?\.(?:\s|$)',
            r'^We should.*?\.(?:\s|$)',
            r'^Thus (?:make|output).*?\.(?:\s|$)',
            r'^We must.*?\.(?:\s|$)',
        ]
        for pattern in thinking_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n', text).strip()
        return text
    
    try:
        # Update Legacy History for UI
        CHAT_HISTORY.append({"role": "user", "content": user_text})

        result = await agent_graph.ainvoke(
            {"messages": [HumanMessage(content=user_text)]},
            config={"configurable": {"thread_id": CURRENT_CHAR["id"]}}
        )
        # Extract the last AI message content
        last_message = result["messages"][-1]
        response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Clean the response
        response_text = clean_response(response_text)
        
        logger.info(f"Agent response: {response_text[:500]}..." if len(response_text) > 500 else f"Agent response: {response_text}")
        if not response_text:
            logger.warning("[AGENT] Empty response after cleaning, returning fallback")
            response_text = "I couldn't find that information. Could you try rephrasing your question?"

        # If memory was cleared by a tool, wipe the checkpointer state for this thread
        # We do this AFTER ainvoke finishes to avoid KeyError mid-run
        if any("Persistent memory file deleted" in str(getattr(m, "content", "")) for m in result.get("messages", [])):
            thread_id = CURRENT_CHAR["id"]
            if hasattr(memory_saver, 'storage'):
                memory_saver.storage.pop((thread_id,), None)
                logger.info(f"[MEMORY] Cleared LangGraph state for thread: {thread_id}")

        # Update Legacy History for UI
        CHAT_HISTORY.append({"role": "assistant", "content": response_text})

        # Async Save
        await save_memory()

        return response_text
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return "I encountered an error processing your request."

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
        
    process = await asyncio.create_subprocess_exec(
        "say",
        "-v", MAC_VOICE,
        "-o", output_file,
        "--data-format=LEF32@22050",
        final_response,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        logger.error(f"Speech synthesis failed (code {process.returncode}): {stderr.decode()}")
        return Response(content="Error generating speech", status_code=500)

    if not os.path.exists(output_file):
         logger.error("Speech synthesis output file missing")
         return Response(content="Error generating speech", status_code=500)

    wav_data = await asyncio.to_thread(read_file_bytes, output_file)

    return Response(content=wav_data, media_type="audio/wav")

if __name__ == "__main__":
    local_ip = get_local_ip()
    port = cfg().port
    logger.info(f"🚀 Server starting on http://{local_ip}:{port}")
    logger.info(f"📱 Connect Enchanted with Base URL: http://{local_ip}:{port}/v1")
    uvicorn.run(app, host=cfg().host, port=port)
