# Setup Instructions

Follow these steps to set up the project on a Mac (Apple Silicon Recommended).

## Hardware Requirements

- **Processor**: Apple Silicon (M1/M2/M3) is strongly recommended for GPU acceleration. Intel Macs will be significantly slower.
- **RAM**: 16GB Minimum (to fit the 8GB model + Whisper + OS).
- **Disk**: ~10GB free space.

## Prerequisites

1. **Homebrew** (Package Manager)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Python 3.10+** (Required)
   ```bash
   brew install python@3.11
   ```

3. **System Libraries**
   Required for audio handling (`portaudio`).
   ```bash
   brew install portaudio ffmpeg
   ```

## Installation

1. **Create Virtual Environment**
   Always run this project in a virtual environment to avoid conflicts.
   ```bash
   cd project-local-voice-character-assistant
   python3.11 -m venv venv
   source venv/bin/activate
   ```

2. **Install Python Dependencies**
   
   **Step 1: Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

   **Step 2: Re-install Llama-cpp-python with Metal Support (IMPORTANT)**
   To ensure the LLM runs fast on your Mac's GPU, you must compile it with Metal support:
   ```bash
   CMAKE_ARGS="-DGGML_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
   ```

## Model Setup

You must download the AI model manually (it's too large for Git).

1. **Download Llama 3**:
   - Go to HuggingFace: [Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF)
   - Download the file named: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` (~4.9GB)
   - Place it in the `project-local-voice-character-assistant/` root directory.

## Configuration (.env)

1. **Gemini API Key (Web Search)**:
   - Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey).
   - Create a file named `.env` in the root folder.
   - Add: `GEMINI_API_KEY=your_key_here`
   - *Note: The free tier has generous rate limits (15 RPM), but if you spam requests, the agent will gracefully fail the search tool.*

## Troubleshooting

- **Microphone Issues**: If `client_v2.py` crashes or can't hear you, ensure your Terminal app has permission to access the Microphone in **System Settings > Privacy & Security > Microphone**.
- **"Process Not Trusted"**: See the README for the Accessibility permissions fix.
