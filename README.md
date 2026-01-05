# Local AI Voice Assistant 🤖

This project enables you to talk to a locally running AI NPC (Non-Player Character) that can hear, think, speak, and remember conversations. It uses a **client-server** architecture where the client captures audio (like a walkie-talkie) and the server processes it using high-performance local AI models.

While it's designed as a playground for AI agents, it's robust enough to act as a daily voice assistant that can search the web and maintain context over time.

## Local-First Design (Privacy & Safety)

This project was built with a "Local-First" philosophy. This is critical for privacy, especially if children are interacting with it.

*   **Your Voice Stays Here**: Audio is transcribed **on-device** using `faster-whisper`. No audio recordings are ever sent to the cloud.
*   **Your Thoughts Stay Here**: The brain (Llama 3) runs entirely on your Mac's GPU (Metal). It doesn't share your conversation with OpenAI or Anthropic.
*   **Safety Layer**: The only external connection is to Google Gemini for factual search queries. We deliberately use Gemini over a raw web scraper because it allows us to enforce **Safety Settings** (blocking harassment, hate speech, and explicit content at the source) before the data even reaches the robot.

## Features

- **Speech-to-Text**: `faster-whisper` provides near-instant local transcription.
- **Brain**: `llama-cpp-python` runs **Llama 3 (8B)** with Metal (GPU) acceleration on macOS.
- **Text-to-Speech**: Uses the native macOS `say` command for zero-latency, character-specific voices.
- **Memory**: Maintains persistent conversation history via JSON files.
- **Modular Engine**: Support for multiple distinct personalities (R2-67, Bender, Zoe) via simple config files.

### Feature Character: R2-67

While the system supports any character, **R2-67** is the default personality.

- **Personality**: Irreverent, clever "R2-67" (Stereotypical Robot with snark) who calls you "meatbag".
- **Voice**: "Zarvox" (Robot voice) or Ralph (fallback).
- **Web Search**: Powered by **Google Gemini 2.5 Flash (Lite)** with grounded search.
  - Unlike pure LLMs, R2-67 can answer questions like *"What is the price of silver?"* or *"Who won the Super Bowl?"* accurately.
- **20 Questions Mode**: R2-67 plays as the guesser in a 20 Questions game.
- **Persistent Memory**: Chat history saved to `robot_memory.json`.
- **Hardware**: Runs comfortably on M1/M2/M3 chips (16GB RAM recommended).

## Architecture & Technical Decisions

This project uses a **Modular Voice Assistant Engine** (`server_voice_assistant.py`) rather than hardcoded scripts. This structure allows you to extend the system with new personalities just by editing a config file.

### 1. Modular Engine Design
*   **Engine**: `server_voice_assistant.py` handles the heavy lifting—Audio I/O, LLM inference, and Tool Calling.
*   **Config**: `characters.py` contains the "Soul" of the agent—System Prompts, Voice selections, and startup ASCII art.
*   **Extensibility**: Adding a new character is as simple as adding a dictionary entry to `characters.py`.

### 2. Stateless vs. Stateful Memory
*   **Legacy**: Early iterations were stateless and forgot context immediately.
*   **Current**: We use a persistent **JSON Memory Store** (unique per character) and a **Sliding Window Context** (last 50 turns). This allows the assistant to reference past topics, play games, and build a relationship with the user.

### 3. JSON Structured Outputs
One of the biggest challenges with LLM agents is reliable tool use.
*   **Problem**: Parsing raw text for commands like `[SEARCH: ...]` is fragile.
*   **Solution**: We use **JSON Structured Outputs** (`response_format={"type": "json_object"}`). The LLM is strictly enforced to output a clear schema. This guarantees that "Thoughts" (internal) are separated from "Speech" (external) and "Actions" (Tools).

### 4. Scoped Push-to-Talk (Client)
Instead of an always-listening wake word (which drains battery and raises privacy concerns), we implemented a **Scoped PTT mechanism** in `client_v2.py`.
*   **Global Hotkey**: The client listens for `SPACE` bar input globally using the Accessibility API.
*   **App Filtering**: To prevent accidental triggers while typing, we utilize an `ALLOWED_APPS` list. The robot *only* listens if you are actively focused on specific apps (like Terminal or VS Code).

### 5. Search Strategy: API vs Scraping
We chose to offload "World Knowledge" to **Gemini 2.5 Flash-Lite**. It is incredibly fast, grounded, and most importantly, allows us to apply robust **Safety Settings** (blocking harmful content) before the data ever reaches our local Llama model.

## Project Structure

```
project-local-voice-character-assistant/
├── server_voice_assistant.py  # MAIN ENGINE: Handles Audio I/O, LLM logic, and Character Loading.
├── characters.py              # CONFIG: Definitions for R2-67, Bender, and Zoe.
├── client_v2.py               # Thin client. Captures audio on keypress -> sends to server.
├── requirements.txt           # Python dependencies (locked to specific safe versions).
├── robot_memory.json          # Persistent brain state for R2-67.
├── .env                       # API keys (not checked in).
├── legacy/                    # Previous iterations (server_robot.py, server.py) for historical context.
└── Meta-Llama-3-*.gguf        # The model weights (downloaded separately).
```

## Setup

See [SETUP.md](SETUP.md) for full installation details.

**Requirements:**
- **Python 3.10+**
- **16GB RAM** (Recommended)
- **Apple Silicon** (M1/M2/M3)

**Quick Start:**
1. Download [Meta-Llama-3-8B-Instruct-Q4 from HuggingFace](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF).
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Gemini API key to `.env` (for web search):
   ```
   GEMINI_API_KEY=your_key_here
   ```
4. Run the assistant: `python server_voice_assistant.py`
5. Select your character from the menu.
6. Run the client: `python client_v2.py`

## Usage

### 1. Start the Server
```bash
python server_voice_assistant.py
```
*You will be asked to choose a character (R2-67, Bender, or Zoe).*

### 2. Start the Client
Open a new terminal window and run:
```bash
python client_v2.py
```
- **Interact**: Hold the **SPACE** bar to talk.
- **Indicator**: A sound prompt will play (Tink/Pop) to indicate recording start/stop.
- **Global Hotkey & Safety**: The client uses the Accessibility API to listen for the hotkey even when the terminal is in the background, but respects the `ALLOWED_APPS` permissions to avoid accidental triggers.

**Troubleshooting - "This process is not trusted" Error:**

If you see this error (macOS security feature):
```
This process is not trusted! Input event monitoring will not be possible...
```

**Fix:**
1. Open **System Settings** → **Privacy & Security** → **Accessibility**
2. Click the **+** button
3. Navigate to `/Applications/Utilities/Terminal.app` (or iTerm2 if you use that)
4. Enable the checkbox
5. Restart the client

### 3. Special Features

**Web Search:**
Ask questions about current events. If the assistant doesn't know, it will trigger a `"call_to_action": "search"`, query Google, and synthesize the answer.

**20 Questions:**
Say "Play 20 questions". The assistant will switch modes to try and guess your object.

## Roadmap

### Immediate Next Steps
*   **Dynamic Sound Effects**: Inject sound effects (beeps, boops) dynamically into the response stream based on context.
*   **Tool Expansion**: Allow the robot to read provided text files or PDFs to gather context before answering.
*   **Code Interpreter**: Give the assistant the ability to run Python code for math or logic puzzles.

### Long Term Ideas
*   **Multi-Device Comm**: Decouple the client to run on a Raspberry Pi or ESP32 (Walkie Talkie form factor) while the Mac acts as the "Brain Server".
*   **Ensemble Cast**: Have multiple agents (R2-67, Bender, Zoe) active in the same chatroom, conversing with each other and the user.
*   **Vision**: Add a camera input to the client so the assistant can see what you are holding.

## Maintenance

**Temporary Files:**
The server generates `temp_output.wav` for audio playback. These are overwritten on each turn.

**Memory Backups:**
Old memory files are simple JSON. If you want to start fresh, just delete `robot_memory.json` (or `bender_memory.json`, etc).
