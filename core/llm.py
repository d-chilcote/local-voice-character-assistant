import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from llama_cpp import Llama
from logger_config import get_logger
from config import cfg

logger = get_logger(__name__)

class BaseLLM(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def create_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 500,
        response_format: Optional[Dict[str, str]] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Unified method to generate a chat completion."""
        pass

class LlamaCPPLLM(BaseLLM):
    """Local LLM implementation using llama-cpp-python."""
    
    def __init__(
        self, 
        model_path: str, 
        n_gpu_layers: int = -1, 
        n_ctx: int = 8192, 
        n_threads: int = 8,
        flash_attn: bool = False,
        use_mlock: bool = False,
        chat_format: Optional[str] = None,
        default_inference_params: Optional[Dict[str, Any]] = None,
        extra_config: Optional[Dict[str, Any]] = None
    ):
        logger.info(f"Initializing LlamaCPP with model: {model_path} (n_ctx: {n_ctx}, flash_attn: {flash_attn}, mlock: {use_mlock})")
        self.default_inference_params = default_inference_params or {}
        self.extra_config = extra_config or {}
        
        # chat_format="jinja" specifically for model templates if needed
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=n_threads,
            flash_attn=flash_attn,
            use_mlock=use_mlock,
            chat_format=chat_format,
            verbose=False,
            **self.extra_config
        )

    def create_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 500,
        response_format: Optional[Dict[str, str]] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        # Priority: kwargs > default_inference_params > method arguments
        params = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "stop": stop
        }
        params.update(self.default_inference_params)
        params.update(kwargs)
        
        response = self.llm.create_chat_completion(**params)
        return response
