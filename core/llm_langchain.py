"""LangChain-compatible LLM wrapper for local GGUF models.

Provides a factory function to create ChatLlamaCpp instances
with consistent configuration for tool-calling support.
"""
from typing import Optional, Dict, Any

from langchain_community.chat_models import ChatLlamaCpp

from logger_config import get_logger

logger = get_logger(__name__)


def create_chat_llm(
    model_path: str,
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,
    n_threads: int = 8,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    flash_attn: bool = True,
    use_mlock: bool = True,
    verbose: bool = False,
    chat_format: Optional[str] = "chatml-function-calling",
    **kwargs
) -> ChatLlamaCpp:
    """
    Factory for ChatLlamaCpp with tool-calling support.
    
    Args:
        model_path: Path to the GGUF model file.
        n_ctx: Context window size.
        n_gpu_layers: Number of layers to offload to GPU (-1 for all).
        n_threads: Number of CPU threads.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate (default 1024).
        flash_attn: Enable flash attention.
        use_mlock: Lock model in memory.
        verbose: Enable verbose output.
        chat_format: Chat format for function calling. Use "chatml-function-calling"
                     for most models, or model-specific formats like "functionary-v2".
        **kwargs: Additional kwargs passed to ChatLlamaCpp.
        
    Returns:
        Configured ChatLlamaCpp instance.
    """
    logger.info(
        f"Creating ChatLlamaCpp: {model_path} "
        f"(n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}, flash_attn={flash_attn}, "
        f"chat_format={chat_format}, max_tokens={max_tokens})"
    )
    
    return ChatLlamaCpp(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        temperature=temperature,
        max_tokens=max_tokens,
        flash_attn=flash_attn,
        use_mlock=use_mlock,
        verbose=verbose,
        chat_format=chat_format,
        **kwargs
    )


