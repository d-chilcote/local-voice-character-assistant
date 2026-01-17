import os
from functools import lru_cache
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

class Config:
    """Centralized configuration for the voice assistant."""
    
    def __init__(self):
        # Paths
        self.llama_path: str = os.getenv("LLAMA_PATH", "./Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")
        
        # API Keys
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        
        # LLM Settings
        self.llama_n_gpu_layers: int = int(os.getenv("LLAMA_GPU_LAYERS", "-1"))
        self.llama_n_ctx: int = int(os.getenv("LLAMA_CTX", "8192"))
        self.llama_n_threads: int = int(os.getenv("LLAMA_THREADS", "8"))
        
        # Chat Settings
        self.history_limit: int = 50
        
        # Server Settings
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))

@lru_cache()
def cfg() -> Config:
    """Returns a cached instance of the configuration."""
    return Config()
