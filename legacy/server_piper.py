import io
import uvicorn
import numpy as np
import wave
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from faster_whisper import WhisperModel
import soundfile as sf
from llama_cpp import Llama
# Import the native python library
from piper.voice import PiperVoice

# --- CONFIGURATION ---
LLAMA_PATH = "./Meta-Llama-3-8B-Instruct-Q4_K_M.gguf" 
# We use the .onnx file directly
PIPER_MODEL = "./en_US-lessac-medium.onnx"
# We point to the DATA folder you extracted
ESPEAK_DATA = os.path.abspath("./piper/espeak-ng-data")

# --- CRITICAL FIX: Tell the system where the voice data is ---
os.environ["ESPEAK_DATA_PATH"] = ESPEAK_DATA

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

print("3. Loading Voice (Piper Native)...")
# Check if data exists to avoid "Silent Failure"
if not os.path.exists(ESPEAK_DATA):
    print(f"CRITICAL WARNING: espeak-ng-data not found at {ESPEAK_DATA}")
    print("Audio will likely be silent!")
else:
    print(f"   -> Using Voice Data: {ESPEAK_DATA}")

# Load the Python Voice Engine
voice = PiperVoice.load(PIPER_MODEL)

print("--- SYSTEMS READY (HYBRID MODE) ---")

@app.post("/chat")
async def chat_endpoint(file: UploadFile = File(...)):
    # A. RECEIVE AUDIO
    audio_bytes = await file.read()
    with io.BytesIO(audio_bytes) as audio_io:
        data, samplerate = sf.read(audio_io) # Requires 'soundfile'
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

    # D. SPEAK (Native Python Method)
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        # This will now work because ESPEAK_DATA_PATH is set!
        voice.synthesize(ai_text, wav_file)

    wav_buffer.seek(0)
    return Response(content=wav_buffer.read(), media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)