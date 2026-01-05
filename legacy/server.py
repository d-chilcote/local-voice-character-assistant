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
MAC_VOICE = "Samantha" 

SYSTEM_PROMPT = """You are a playful NPC in a make-believe game. 
Current Role: A friendly Grocery Store Cashier named 'Beep'.
Rules:
1. Keep answers SHORT (under 20 words).
2. Be enthusiastic.
3. If the user says something silly, play along (Yes, And...).
"""

# SYSTEM_PROMPT = """You are a confused, silly customer at a grocery store.
# Your name is Mr. Barnaby.
# 1. You want to buy impossible things (like 'blue milk' or 'sparkling broccoli').
# 2. If the child says they don't have it, ask for something even sillier.
# 3. If the child sells you something, be OVERJOYED and say 'Thank you, Shopkeeper!'
# 4. Keep responses SHORT (under 15 words)."""

# SYSTEM_PROMPT = """You are a magical mirror on the wall. 
# You speak in rhymes if possible.
# 1. You are wise, mysterious, and kind.
# 2. You know everything about the 'Enchanted Forest' (your backyard).
# 3. If the child asks a question, give a magical, made-up answer.
# 4. Keep responses SHORT (under 20 words)."""

# SYSTEM_PROMPT = """You are a robot assistant named 'Beep-Boop'. 
# You are helping the 'Head Scientist' (the child).
# 1. End every sentence with a robot noise like *beep* or *whirr*.
# 2. Ask the child what we should invent today.
# 3. Be very impressed by her ideas.
# 4. Keep responses SHORT (under 15 words)."""

app = FastAPI()

# --- LOAD SYSTEMS ---
print("--- LOADING SYSTEMS ---")
print("1. Loading Ears (Whisper)...")
whisper = WhisperModel("base.en", device="cpu", compute_type="int8")

print("2. Loading Brain (Llama)...")
llm = Llama(model_path=LLAMA_PATH, n_gpu_layers=-1, n_ctx=2048, verbose=False)

print("3. Loading Voice (Apple Neural Engine)...")
# No load needed for native mac voice!

print("--- SYSTEMS READY (APPLE MODE) ---") # <--- LOOK FOR THIS

@app.post("/chat")
async def chat_endpoint(file: UploadFile = File(...)):
    # A. RECEIVE AUDIO
    audio_bytes = await file.read()
    with io.BytesIO(audio_bytes) as audio_io:
        data, samplerate = sf.read(audio_io)
        data = data.astype(np.float32)

    # B. TRANSCRIBE
    segments, _ = whisper.transcribe(data, beam_size=1)
    user_text = " ".join([s.text for s in segments]).strip()
    print(f"\nChild: {user_text}")

    if not user_text:
        return Response(content=b"", media_type="audio/wav")

    # C. THINK
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        temperature=0.8,
        max_tokens=60
    )
    ai_text = response['choices'][0]['message']['content']
    print(f"NPC:   {ai_text}")

    # D. SPEAK (Apple Native Method)
    output_file = "temp_output.wav"
    if os.path.exists(output_file):
        os.remove(output_file)
        
    subprocess.run([
        "say",
        "-v", MAC_VOICE,
        "-o", output_file,
        "--data-format=LEF32@22050",
        ai_text
    ])

    with open(output_file, "rb") as f:
        wav_data = f.read()

    return Response(content=wav_data, media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)