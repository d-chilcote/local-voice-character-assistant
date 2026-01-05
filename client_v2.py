"""
Scoped Push-to-Talk Client (v2.0)
=================================
A privacy-focused "Walkie Talkie" client for the Voice Assistant.

Features:
  1. Global Hotkey: Listens for SPACE bar (via pynput).
  2. Scoped Focus:  Only records if an "Allowed App" (e.g., Terminal, VSCode) is active.
                    This prevents accidental triggers while typing in other apps.
  3. Audio Capture: Buffered reading via sounddevice -> WAV.
  4. Server Comm:   POSTs WAV to the local server, plays back response.
"""

import sounddevice as sd
import numpy as np
import requests
import io
import soundfile as sf
from pynput import keyboard
import os
import queue
import sys
import threading
import time
import subprocess

# --- CONFIG ---
SERVER_URL = "http://localhost:8000/chat"
SAMPLE_RATE = 16000
CHANNELS = 1

# Apps allowed to trigger the Walkie Talkie
ALLOWED_APPS = ["Terminal", "iTerm2", "Code", "Visual Studio Code", "Python", "PyCharm"]

# --- GLOBAL STATE ---
SAFE_TO_RECORD = False 
CURRENT_APP_NAME = "Unknown"

task_queue = queue.Queue()

def get_active_app_name():
    """Uses AppleScript to safely get the frontmost app name"""
    try:
        # This command asks macOS System Events for the active app name
        cmd = "osascript -e 'tell application \"System Events\" to get name of first application process whose frontmost is true'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "Unknown"

def window_watcher():
    """Background thread that checks the active window every 0.5s"""
    global SAFE_TO_RECORD, CURRENT_APP_NAME
    last_app = ""
    
    print("[SYSTEM] Window Watcher started...")
    
    while True:
        # 1. Get the App Name safely
        app_name = get_active_app_name()
        
        # 2. Only print if it changed (so we don't spam logs)
        if app_name != last_app:
            print(f"[FOCUS] Switched to: '{app_name}'")
            if app_name in ALLOWED_APPS:
                print("   -> Walkie Talkie ENABLED (Green Light)")
            else:
                print("   -> Walkie Talkie DISABLED (Red Light)")
            last_app = app_name

        # 3. Update the Global Variable
        CURRENT_APP_NAME = app_name
        if app_name in ALLOWED_APPS:
            SAFE_TO_RECORD = True
        else:
            SAFE_TO_RECORD = False
            
        # Check twice a second
        time.sleep(0.5)

class WalkieTalkie:
    def __init__(self):
        self.recording = False
        self.audio_buffer = []
        self.stream = None

    def on_press(self, key):
        if key == keyboard.Key.space and not self.recording:
            # Check the Green Light
            if not SAFE_TO_RECORD:
                return 

            # Start Recording
            os.system("afplay /System/Library/Sounds/Tink.aiff&")
            print(f"\n[REC] Listening in {CURRENT_APP_NAME}... (Release SPACE to send)")
            
            self.recording = True
            self.audio_buffer = []
            self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self.audio_callback)
            self.stream.start()

    def on_release(self, key):
        if key == keyboard.Key.space and self.recording:
            os.system("afplay /System/Library/Sounds/Pop.aiff&")
            print("[STOP] Thinking...")
            
            self.recording = False
            self.stream.stop()
            self.stream.close()
            
            audio_copy = list(self.audio_buffer)
            task_queue.put(audio_copy)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, flush=True)
        self.audio_buffer.append(indata.copy())

def main_loop():
    print("--- MAC WALKIE TALKIE (AppleScript Mode) ---")
    print(f"Allowed Apps: {ALLOWED_APPS}")
    print("Hold SPACEBAR to talk (Only works when allowed app is focused).")
    print("Press CTRL+C to quit.")

    # 1. Start the Window Watcher
    watcher = threading.Thread(target=window_watcher, daemon=True)
    watcher.start()

    # 2. Start the Keyboard Listener
    client = WalkieTalkie()
    listener = keyboard.Listener(on_press=client.on_press, on_release=client.on_release)
    listener.start()

    # 3. Main Loop
    while True:
        try:
            audio_chunks = task_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if not audio_chunks:
            continue

        try:
            print("[NET] Sending to AI...")
            audio_data = np.concatenate(audio_chunks, axis=0)
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_data, SAMPLE_RATE, format='WAV')
            wav_io.seek(0)

            response = requests.post(SERVER_URL, files={"file": ("input.wav", wav_io, "audio/wav")})
            
            if response.status_code == 200:
                print(f"[RCV] Received {len(response.content)} bytes")
                with open("npc_response.wav", "wb") as f:
                    f.write(response.content)
                
                print("[PLAY] Playing audio...")
                os.system("afplay npc_response.wav")
                print("\n[RDY] Press SPACE to talk")
            else:
                print(f"Error: {response.status_code}")
        
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit()