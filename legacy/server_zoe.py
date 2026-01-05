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
# The exact name you tested in the terminal
MAC_VOICE = "Zoe" 

SYSTEM_PROMPT = """You are a playful NPC in a make-believe game. 
Current Role: A friendly Grocery Store Cashier named 'Beep'.
Rules:
1. Keep answers SHORT (under 20 words).
2. Be enthusiastic.
3. If the user says something silly, play along (Yes, And...).
"""

app = FastAPI()

# --- LOAD SYSTEMS ---
print("--- LOADING SYSTEMS ---")
print("1. Loading Ears (Whisper)...")
whisper = WhisperModel("base.en", device="cpu", compute_type="int8")

print("2. Loading Brain (Llama)...")
llm = Llama(model_path=LLAMA_PATH, n_gpu_layers=-1, n_ctx=2048, verbose=False)

print(f"3. Voice Engine: Apple Neural ({MAC_VOICE})...")
print("--- SYSTEMS READY (ZOE MODE) ---")

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

    # D. SPEAK (Apple Native)
    output_file = "temp_output.wav"
    if os.path.exists(output_file):
        os.remove(output_file)
        
    cmd = [
        "say",
        "-v", MAC_VOICE,
        "-r", "140",  # <--- NEW: Sets speed to 140 words per minute
        "-o", output_file,
        "--data-format=LEF32@22050",
        ai_text
    ]
    
    # Run the command and capture errors if it fails
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Voice Error: {result.stderr}")
        # Fallback to default voice if Zoe fails
        subprocess.run(["say", "-o", output_file, ai_text])

    # Send audio back to client
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            wav_data = f.read()
        return Response(content=wav_data, media_type="audio/wav")
    else:
        return Response(content=b"", media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)