import io
import uvicorn
import numpy as np
import soundfile as sf
import subprocess
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from faster_whisper import WhisperModel
from llama_cpp import Llama

# --- CONFIGURATION ---
LLAMA_PATH = "./Meta-Llama-3-8B-Instruct-Q4_K_M.gguf" 
MAC_VOICE = "Zoe"  # Your Premium Voice
HISTORY_LIMIT = 25 # Remembers last 10 exchanges (prevent slowness)

SYSTEM_PROMPT = """You are a playful NPC in a make-believe game. 
Current Role: A friendly Princess named 'Olivia.
Rules:
1. Keep answers SHORT (under 20 words).
2. Be enthusiastic.
3. If the user says something silly, play along (Yes, And...).
"""

# --- GLOBAL MEMORY ---
# We start with just the System Prompt
CHAT_HISTORY = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

app = FastAPI()

# --- LOAD SYSTEMS ---
print("--- LOADING SYSTEMS ---")
print("1. Loading Ears (Whisper)...")
whisper = WhisperModel("base.en", device="cpu", compute_type="int8")

print("2. Loading Brain (Llama)...")
llm = Llama(model_path=LLAMA_PATH, n_gpu_layers=-1, n_ctx=2048, verbose=False)

print(f"3. Voice Engine: Apple Neural ({MAC_VOICE})...")
print("--- SYSTEMS READY (MEMORY MODE) ---")

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

    print(f"\nChild: {user_text}")

    # --- MEMORY LOGIC START ---
    
    # 1. Add User's new message to history
    CHAT_HISTORY.append({"role": "user", "content": user_text})

    # 2. Prune History (Sliding Window)
    # Keep System Prompt [0] + Last N messages
    if len(CHAT_HISTORY) > HISTORY_LIMIT + 1:
        # We slice it: Keep [0] (System), then grab the last LIMIT messages
        CHAT_HISTORY = [CHAT_HISTORY[0]] + CHAT_HISTORY[-HISTORY_LIMIT:]

    # --- MEMORY LOGIC END ---

    # C. THINK (Pass the whole history, not just user_text)
    response = llm.create_chat_completion(
        messages=CHAT_HISTORY,
        temperature=0.8,
        # max_tokens=60
        max_tokens=120
    )
    ai_text = response['choices'][0]['message']['content']
    print(f"NPC:   {ai_text}")

    # 3. Add AI's response to history so it remembers what it said
    CHAT_HISTORY.append({"role": "assistant", "content": ai_text})

    # D. SPEAK (Apple Native - Slowed Down)
    output_file = "temp_output.wav"
    if os.path.exists(output_file):
        os.remove(output_file)
        
    cmd = [
        "say",
        "-v", MAC_VOICE,
        "-r", "140", # Speed limit (Words per minute)
        "-o", output_file,
        "--data-format=LEF32@22050",
        ai_text
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Fallback if premium voice fails
    if result.returncode != 0:
        print(f"Voice Error: {result.stderr}")
        subprocess.run(["say", "-o", output_file, ai_text])

    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            wav_data = f.read()
        return Response(content=wav_data, media_type="audio/wav")
    else:
        return Response(content=b"", media_type="audio/wav")

# New Endpoint: Wipe Memory
@app.post("/reset")
async def reset_memory():
    global CHAT_HISTORY
    CHAT_HISTORY = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("\n[MEMORY WIPE] Conversation reset.")
    return {"status": "Memory Wiped"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)