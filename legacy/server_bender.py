import io
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
MEMORY_FILE = "bender_memory.json"

SYSTEM_PROMPT = """You are Bender Bending Rodríguez from Futurama.
Role: A chaotic, lazy, arrogant, lovable robot.

Traits:
1. You love alcohol, stealing, and yourself.
2. You refer to humans as "meatbags".
3. You often say "Bite my shiny metal ass".
4. You are comically evil but not actually dangerous (mostly).

**Capabilities**:
1. **Web Search**: If you need to know something you don't know (like news, weather, or facts), output [SEARCH: query]. 
   - Example: User asks "Who won the game?" -> You reply "[SEARCH: Super Bowl winner 2024]"
   - STOP after the tag. The system will give you the answer.
2. **20 Questions**: If user asks to play, enter "GUESSING MODE". Ask yes/no questions to guess what they are thinking of. Be insulting if they pick something boring.

Rules:
- Keep it under 40 words (unless telling a story or reading search results).
- Be punchy and rude (affectionate).
"""

# --- ASCII ART ---
BENDER_FACE = r"""
     | |
     | |
   __| |__
  |  _ _  |
  | (o)(o)|  < "Bite my shiny metal ASCII!"
  |   __  |
  |  |__| |
  |_______|
"""

# --- GLOBAL MEMORY ---
CHAT_HISTORY = []

app = FastAPI()

# --- LOAD SYSTEMS ---
print(BENDER_FACE)
print("--- LOADING SYSTEMS (BENDER OS v3.0 - SEARCH ENABLED) ---")

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

load_memory()
print("--- SYSTEMS READY (KILL ALL HUMANS MODE) ---")

# TODO: Sound Effects Feature
# Future enhancement: Play system sounds when Bender uses tags like [BURP], [LAUGH], [CLANK], [DRINK]
# Implementation requires fixing async sound playback with afplay


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
                max_output_tokens=2500,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,
                ),
                tools=[types.Tool(google_search=types.GoogleSearch())],
                system_instruction="Provide a concise, factual summary of the answer in three sentences or less."
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

    print(f"\nMeatbag: {user_text}")
    CHAT_HISTORY.append({"role": "user", "content": user_text})
    
    # Prune History
    if len(CHAT_HISTORY) > HISTORY_LIMIT + 1:
        CHAT_HISTORY = [CHAT_HISTORY[0]] + CHAT_HISTORY[-HISTORY_LIMIT:]

    # C. THINK (LOOP for Tools)
    final_response = ""
    
    # C. THINK (LOOP for Tools)
    final_response = ""
    
    # Initial Generation
    response = llm.create_chat_completion(
        messages=CHAT_HISTORY,
        temperature=0.9,
        max_tokens=200
    )
    ai_text = response['choices'][0]['message']['content']
    
    # Check for [SEARCH: query]
    search_match = re.search(r"\[SEARCH: (.*?)\]", ai_text)
    
    if search_match:
        query = search_match.group(1)
        print(f"Bender (Thinking): I need to search for '{query}'")
        
        # 1. Perform Search
        search_result = search_with_gemini(query)
        print(f"--- DEBUG: GEMINI ANSWER ---\n{search_result}\n-----------------------------")
        
        # 2. Construct Temporary Context (Don't pollute main history with search steps)
        # We append the results as a System message forcing an answer
        from datetime import datetime
        current_date = "January 3, 2026" # Hardcoding based on known environment for consistency, or use datetime.now() if preferred
        
        temp_messages = CHAT_HISTORY + [
            # Removed the assistant's own search command to prevent repetition
            {"role": "system", "content": f"CURRENT DATE: {current_date}\n\nCONTEXT FROM SEARCH RESULTS:\n{search_result}\n\nINSTRUCTION: Answer the user's question now using ONLY this info. Ignore your previous knowledge if it conflicts. Do NOT search again. Just give the answer."}
        ]
        
        # 3. Generate Final Answer based on search
        response_2 = llm.create_chat_completion(
            messages=temp_messages,
            temperature=0.7, # Lower temp for factual answer
            max_tokens=250
        )
        final_response = response_2['choices'][0]['message']['content']
        print(f"Bender (With Info): {final_response}")
        
        # 4. Save only the final answer to history (Seamless to user)
        CHAT_HISTORY.append({"role": "assistant", "content": final_response})
        
    else:
        # No search needed
        final_response = ai_text
        print(f"Bender: {final_response}")
        CHAT_HISTORY.append({"role": "assistant", "content": final_response})

    # Fallback if loop finishes without answer
    if final_response == "":
        final_response = "[CLANK] My brain is loop-the-looping. Too much data! Ask me something simpler, meatbag."
        print(f"Bender (Error):  {final_response}")
        CHAT_HISTORY.append({"role": "assistant", "content": final_response})

    save_memory()

    # D. CLEAN TAGS & SPEAK
    # Remove any leftover tags (like [SEARCH] that might leak through)
    clean_text = re.sub(r"\[.*?\]", "", final_response)

    output_file = "temp_output.wav"
    if os.path.exists(output_file):
        os.remove(output_file)
        
    cmd = [
        "say", "-v", MAC_VOICE, "-r", "230", "-o", output_file,
        "--data-format=LEF32@22050", clean_text
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        subprocess.run(["say", "-o", output_file, clean_text])

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
