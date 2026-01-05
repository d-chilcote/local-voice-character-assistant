"""
Modular Voice Assistant Server (v2.0)
=====================================
The central nervous system for the Local AI Voice Assistant.

Architecture:
  1. Audio Input:  Received as WAV via /chat endpoint (from client_v2.py)
  2. Ears (STT):   faster-whisper (local, Metal optimized)
  3. Brain (LLM):  Llama 3 8B via llama-cpp-python (local, Metal optimized)
  4. Tool Use:     Google Gemini 2.5 Flash-Lite (API) for grounded search
  5. Voice (TTS):  macOS 'say' command (Zero latency)

Key Features:
  - Component Architecture: Decoupled logic from Character Configs (characters.py)
  - Safety First: Search results are filtered via Gemini Safety Settings *before* reaching the LLM.
  - Testable: Core logic extracted into pure functions.
"""

import io
import sys
import uvicorn
import numpy as np
import soundfile as sf
import subprocess
import os
import json
import re
import asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from faster_whisper import WhisperModel
from llama_cpp import Llama
from google import genai
from google.genai import types
from dotenv import load_dotenv

from characters import CHARACTERS

load_dotenv()

# --- CONFIGURATION (Load Defaults) ---
LLAMA_PATH = "./Meta-Llama-3-8B-Instruct-Q4_K_M.gguf" 
CURRENT_CHAR = None # Will be set by CLI selection
CHAT_HISTORY = []

# --- CLI CHARACTER SELECTION ---
def select_character():
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
    # If running directly, ask for character
    selected = select_character()
    CURRENT_CHAR = selected
else:
    # If imported (unlikely for Uvicorn), default to R2-67
    CURRENT_CHAR = CHARACTERS[0] 

# --- DERIVED CONFIG ---
MEMORY_FILE = CURRENT_CHAR["memory_file"]
SYSTEM_PROMPT = CURRENT_CHAR["system_prompt"]
HISTORY_LIMIT = 50

# --- VOICE CONFIG ---
# Try Native Voice, Fallback if missing
def get_voice():
    target = CURRENT_CHAR["voice_native"]
    if subprocess.run(["say", "-v", target, "test"], capture_output=True).returncode == 0:
        return target
    print(f"[VOICE] '{target}' not found. Falling back to '{CURRENT_CHAR['voice_fallback']}'.")
    return CURRENT_CHAR["voice_fallback"]

MAC_VOICE = get_voice()

# --- ASCII ART ---
print(CURRENT_CHAR["face"])
print(f"--- LOADING {CURRENT_CHAR['name']} (v2.0 MODULAR) ---")
print(f"Memorizing to: {MEMORY_FILE}")


app = FastAPI()

# --- LOAD SYSTEMS ---
print("1. Loading Ears (Whisper)...")
whisper = WhisperModel("base.en", device="cpu", compute_type="int8")

print("2. Loading Brain (Llama)...")
# Optimized for M2 Ultra (64GB)
llm = Llama(model_path=LLAMA_PATH, n_gpu_layers=-1, n_ctx=8192, n_threads=8, verbose=False)

print(f"3. Voice Engine: {MAC_VOICE}...")

# --- PERSISTENCE ---
def load_memory():
    global CHAT_HISTORY
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                CHAT_HISTORY = json.load(f)
            print(f"[MEMORY] Loaded {len(CHAT_HISTORY)} previous thoughts.")
        except Exception as e:
            print(f"[MEMORY] Error loading memory: {e}")
            CHAT_HISTORY = [{"role": "system", "content": SYSTEM_PROMPT}]
    else:
        CHAT_HISTORY = [{"role": "system", "content": SYSTEM_PROMPT}]

def save_memory():
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(CHAT_HISTORY, f, indent=2)
    except Exception as e:
        print(f"[MEMORY] Error saving memory: {e}")

# --- CLI FLAGS ---
if "--reset" in sys.argv:
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
        print("[MEMORY] Wiped memory due to --reset flag.")

load_memory()
print("--- SYSTEMS READY ---")


# --- PURE FUNCTIONS (TESTABLE) ---
def search_with_gemini(query):
    """
    Performs a grounded Google Search via Gemini 2.5 Flash-Lite.
    
    Why API?
    We use the API instead of a raw scraper to enforce strict Safety Settings.
    This ensures that Hate Speech, Harassment, and Dangerous Content are blocked
    at the source before the data ever enters the local LLM context.
    """
    print(f"[SEARCHING via Gemini v2.5] '{query}'...")
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        model = "gemini-flash-lite-latest"
        
        prompt = f"Using Google Search, find the answer to this query: '{query}'."
        
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1000, 
                thinking_config=types.ThinkingConfig(
                    thinking_budget=1024,
                ),
                tools=[types.Tool(google_search=types.GoogleSearch())],
                system_instruction="Provide a concise, factual answer.",
                 safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_LOW_AND_ABOVE", 
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_LOW_AND_ABOVE", 
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_LOW_AND_ABOVE", 
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_LOW_AND_ABOVE", 
                    ),
                ],
            )
        )
        
        result_text = response.text
        print(f"[GEMINI RESULTS] Length: {len(result_text)}")
        return result_text
        
    except Exception as e:
        print(f"[SEARCH ERROR] {e}")
        return f"Search failed: {e}"

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

    print(f"\nUser: {user_text}")

    # C. UPDATE HISTORY (User)
    CHAT_HISTORY.append({"role": "user", "content": user_text})
    if len(CHAT_HISTORY) > HISTORY_LIMIT + 1:
        CHAT_HISTORY = [CHAT_HISTORY[0]] + CHAT_HISTORY[-HISTORY_LIMIT:]

    # D. THINK (Step 1: Plan)
    response = llm.create_chat_completion(
        messages=CHAT_HISTORY,
        temperature=0.7,
        max_tokens=300,
        response_format={"type": "json_object"},
        stop=["}"] # CRITICAL SAFETY against partial JSON
    )

    raw_json = response['choices'][0]['message']['content']
    # Safety: Ensure it ends with bracket if cutoff
    if not raw_json.strip().endswith("}"): raw_json += "}"

    try:
        ai_data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}\nRaw: {raw_json}")
        ai_data = {"thought": "Error", "call_to_action": "reply", "speech": "My logic circuits are jamming."}

    thought = ai_data.get("thought")
    action = ai_data.get("call_to_action")
    speech = ai_data.get("speech")

    print(f"{CURRENT_CHAR['name']} (Thought): {thought}")
    
    final_response = speech

    # E. ACT (Search Loop)
    if action == "search":
        query = ai_data.get("search_query")
        print(f"{CURRENT_CHAR['name']} (Action): Searching for '{query}'...")
        
        # Perform Search
        search_result = search_with_gemini(query)
        
        # Inject Context into LAST User Message (Llama 3 Rule)
        from datetime import datetime
        current_date = "January 3, 2026"
        
        temp_messages = list(CHAT_HISTORY)
        if temp_messages and temp_messages[-1]['role'] == 'user':
            last_msg = temp_messages.pop()
            new_content = f"{last_msg['content']}\n\n[SYSTEM: SEARCH RESULTS]\nDATE: {current_date}\nDATA:\n{search_result}\n\n[INSTRUCTION]: Answer based on the data above. Output JSON with action='reply'."
            temp_messages.append({"role": "user", "content": new_content})
            
        # Re-Generate with Context
        response_2 = llm.create_chat_completion(
            messages=temp_messages,
            temperature=0.7,
            max_tokens=300,
            response_format={"type": "json_object"},
            stop=["}"]
        )
        raw_2 = response_2['choices'][0]['message']['content']
        if not raw_2.strip().endswith("}"): raw_2 += "}"
        
        try:
            ai_data_2 = json.loads(raw_2)
            final_response = ai_data_2.get("speech", "Error generating speech.")
            print(f"{CURRENT_CHAR['name']} (With Info): {final_response}")
        except json.JSONDecodeError as e:
             print(f"JSON Error (Post-Search): {e}\nRaw: {raw_2}")
             final_response = "I found it, but I can't read. Ask me again."
        
    else:
        # Default Reply
        print(f"{CURRENT_CHAR['name']}: {final_response}")

    # 3. SAVE HISTORY 
    CHAT_HISTORY.append({"role": "assistant", "content": final_response})
    save_memory()

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
