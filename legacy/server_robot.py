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

load_dotenv()

# --- CONFIGURATION ---
LLAMA_PATH = "./Meta-Llama-3-8B-Instruct-Q4_K_M.gguf" 
MAC_VOICE = "Zarvox" # The classic Robot voice
if not subprocess.run(["say", "-v", "Zarvox", "test"], capture_output=True).returncode == 0:
    MAC_VOICE = "Ralph" 

HISTORY_LIMIT = 50 
MEMORY_FILE = "robot_memory.json"

SYSTEM_PROMPT = """You are R2-67, an irreverent, clever AI robot.
Role: A witty, self-aware AI who finds humans amusing.

**OUTPUT FORMAT**: You MUST output valid JSON ONLY. No preamble.
Schema:
{
  "thought": "Internal reasoning about the user request.",
  "call_to_action": "search" | "reply" | "none",
  "search_query": "The search query (if action is search, else null)", 
  "speech": "What you want to say to the user (if action is reply, else null)"
}

**Rules**:
1. **Knowledge Cutoff (2023)**: You DO NOT know current events/stats. 
2. **Search**: If asked for ANY list, statistic, news, or fact you aren't 100% sure of, output `call_to_action: "search"` and a `search_query`.
3. **Speech**: This is what the user hears. Be snarky ("meatbags"), clear, and concise.
4. **No Hallucination**: Do not make up facts. Search first.
"""

# --- ASCII ART ---
ROBOT_FACE = r"""
    ╔════════╗
    ║ ◉    ◉ ║
    ║   ▼    ║  < "Another meatbag needs my help?"
    ║  ════  ║
    ╚════════╝
"""

# --- GLOBAL MEMORY ---
CHAT_HISTORY = []

app = FastAPI()

# --- LOAD SYSTEMS ---
print(ROBOT_FACE)
print("--- LOADING SYSTEMS (ROBOT OS v1.0 - SEARCH ENABLED) ---")

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
print("--- SYSTEMS READY (PREPARE FOR SARCASM) ---")

# TODO: Sound Effects Feature
# Future enhancement: Play system sounds for robot expression tags
# Implementation requires fixing async sound playback with afplay


from typing import Optional

# --- PURE FUNCTIONS (TESTABLE) ---
def extract_search_query(text: str) -> str | None:
    """Extracts search query from text, handling various formats."""
    # Check for [SEARCH: query] or SEARCH: query
    # Regex improvement: match "SEARCH:" followed by any text until ] or end of string
    search_match = re.search(r"\[?SEARCH:\s*([^\]]*)(?:\]|$)", text, re.IGNORECASE)
    
    if search_match and search_match.group(1).strip():
        return search_match.group(1).strip()
    return None

def clean_response_text(text: str) -> str:
    """Removes special tags from text."""
    return re.sub(r"\[.*?\]", "", text)

def update_chat_history(history: list, role: str, content: str, limit: int) -> list:
    """Updates history with new message and enforces sliding window limit."""
    history.append({"role": role, "content": content})
    # Prune History (keep system prompt at index 0)
    if len(history) > limit + 1:
        return [history[0]] + history[-limit:]
    return history


def search_with_gemini(query):
    print(f"[SEARCHING via Gemini v1.0] '{query}'...")
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        model = "gemini-flash-lite-latest"
        
        prompt = f"Using Google Search, find the answer to this query: '{query}'."
        
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1000, # Reduced to save cost
                thinking_config=types.ThinkingConfig(
                    thinking_budget=1024, # Enable Thinking for better accuracy
                ),
                tools=[types.Tool(google_search=types.GoogleSearch())],
                system_instruction="Provide a concise, factual answer. If the user asks for a list or statistics, provide the specific items and numbers clearly. Do not truncate lists.",
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_LOW_AND_ABOVE",  # Block most
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_LOW_AND_ABOVE",  # Block most
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_LOW_AND_ABOVE",  # Block most
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_LOW_AND_ABOVE",  # Block most
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
    CHAT_HISTORY = update_chat_history(CHAT_HISTORY, "user", user_text, HISTORY_LIMIT)

    # C. THINK (LOOP for Tools)
    final_response = ""
    
    # 1. INITIAL GENERATION (JSON Mode)
    try:
        response = llm.create_chat_completion(
            messages=CHAT_HISTORY,
            temperature=0.7,
            max_tokens=300,
            response_format={"type": "json_object"},
            stop=["}"] # Stop after closing JSON object
        )
        raw_content = response['choices'][0]['message']['content']
        if not raw_content.strip().endswith("}"): 
            raw_content += "}" # Auto-close if truncated
            
        ai_data = json.loads(raw_content)
        print(f"R2-67 (Thought): {ai_data.get('thought')}")
        
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        final_response = "My circuits are garbled. JSON error."
        ai_data = {"call_to_action": "reply", "speech": final_response}

    # 2. ACTION HANDLER
    action = ai_data.get("call_to_action", "reply")
    
    if action == "search":
        query = ai_data.get("search_query")
        print(f"R2-67 (Action): Searching for '{query}'...")
        
        # Perform Search
        search_result = search_with_gemini(query)
        
        # Inject Context into LAST User Message (Llama 3 Rule)
        # This tricks the model into thinking the user provided the context
        from datetime import datetime
        current_date = "January 3, 2026"
        
        temp_messages = list(CHAT_HISTORY)
        if temp_messages and temp_messages[-1]['role'] == 'user':
            last_msg = temp_messages.pop()
            new_content = f"{last_msg['content']}\n\n[SYSTEM: SEARCH RESULTS]\nDATE: {current_date}\nDATA:\n{search_result}\n\n[INSTRUCTION]: Answer the user based on the data above. Output JSON with action='reply'."
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
        
        if not raw_2.strip().endswith("}"): raw_2 += "}"
        
        try:
            ai_data_2 = json.loads(raw_2)
            final_response = ai_data_2.get("speech", "Error generating speech.")
            print(f"R2-67 (With Info): {final_response}")
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse search response JSON: {e}")
            print(f"RAW OUTPUT: {raw_2}")
            final_response = "I found the info, but I'm having trouble processing it mentally. Ask me again."
        
    else:
        # Default Reply
        final_response = ai_data.get("speech", "Beep boop.")
        print(f"R2-67: {final_response}")

    # 3. SAVE HISTORY (Only the speech, keeping the illusion of conversation)
    # We save as 'assistant' so the history looks normal to the LLM next turn
    CHAT_HISTORY = update_chat_history(CHAT_HISTORY, "assistant", final_response, HISTORY_LIMIT)
    save_memory()

    # D. SPEAK
    output_file = "temp_output.wav"
    if os.path.exists(output_file):
        os.remove(output_file)
        
    cmd = [
        "say", "-v", MAC_VOICE, "-r", "230", "-o", output_file,
        "--data-format=LEF32@22050", final_response
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        subprocess.run(["say", "-o", output_file, final_response])

    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            wav_data = f.read()
        return Response(content=wav_data, media_type="audio/wav")
    else:
        return Response(content=b"", media_type="audio/wav")

@app.post("/reset")
async def reset_memory():
    global CHAT_HISTORY
    CHAT_HISTORY = [{"role": "system", "content": SYSTEM_PROMPT}]
    save_memory()
    print("\n[MEMORY WIPE] I forgot everything! woo!")
    return {"status": "Memory Wiped"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
