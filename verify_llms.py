import sys
import os
import requests
from core.llm import LlamaCPPLLM, OllamaLLM
from config import cfg

def test_llm_abstraction():
    print("--- Testing LLM Abstraction Layer ---")
    
    # 1. Test LlamaCPP (if model exists)
    if os.path.exists(cfg().llama_path):
        print("\n[LlamaCPP] Initializing...")
        try:
            llm = LlamaCPPLLM(
                model_path=cfg().llama_path,
                n_gpu_layers=0, # Use CPU for quick test
                n_ctx=512
            )
            messages = [{"role": "user", "content": "Say hello world"}]
            print("[LlamaCPP] Generating...")
            resp = llm.create_chat_completion(messages, max_tokens=10)
            print(f"[LlamaCPP] Response: {resp['choices'][0]['message']['content']}")
        except Exception as e:
            print(f"[LlamaCPP] ERROR: {e}")
    else:
        print("\n[LlamaCPP] SKIPPED: Model file not found.")

    # 2. Test Ollama (if running)
    print("\n[Ollama] Checking connection...")
    try:
        resp = requests.get(f"{cfg().ollama_base_url}/api/tags", timeout=1)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if models:
                model_name = models[0]["name"]
                print(f"[Ollama] Initializing with: {model_name}")
                llm = OllamaLLM(model_name=model_name, base_url=cfg().ollama_base_url)
                messages = [{"role": "user", "content": "Say hello world"}]
                print("[Ollama] Generating...")
                resp = llm.create_chat_completion(messages, max_tokens=10)
                print(f"[Ollama] Response: {resp['choices'][0]['message']['content']}")
            else:
                print("[Ollama] No models found in Ollama.")
        else:
            print(f"[Ollama] API returned status {resp.status_code}")
    except Exception as e:
        print(f"[Ollama] SKIPPED: Could not connect to Ollama ({e})")

if __name__ == "__main__":
    test_llm_abstraction()
