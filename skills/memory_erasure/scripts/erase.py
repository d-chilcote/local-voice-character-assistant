import os
import logging
from typing import List, Dict, Any

# Standard project logger
logger = logging.getLogger(__name__)

def execute(history: List[Dict[str, str]], memory_file: str, **kwargs) -> str:
    """
    Erases the agent's memory file and clears the in-memory history.
    
    Args:
        history: The mutable list of conversation history.
        memory_file: Path to the persistent JSON memory file.
        
    Returns:
        Status message string.
    """
    logger.info(f"Initiating memory erasure. File: {memory_file}")
    
    status_msg = []
    
    # 1. Delete Persistent File
    if memory_file and os.path.exists(memory_file):
        try:
            os.remove(memory_file)
            status_msg.append("Persistent memory file deleted.")
        except Exception as e:
            logger.error(f"Failed to delete memory file: {e}")
            status_msg.append(f"Error deleting file: {e}")
    else:
        status_msg.append("No persistent memory file found to delete.")
        
    # 2. Clear In-Memory History
    # We want to preserve the *current* turn's context so the agent can reply "I forgot everything."
    # AND we must preserve the system prompt (usually at index 0).
    
    if not history:
        status_msg.append("History was already empty.")
        return "Memory erasure complete (no history found)."

    # Locate System Prompt
    system_prompt_item = None
    if history and history[0].get("role") == "system":
        system_prompt_item = history[0]

    # Locate Trigger Message (last user message)
    # This ensures the agent has context to reply to the "erase memory" command
    trigger_message = None
    if history and history[-1].get("role") == "user":
        trigger_message = history[-1]
        
    # Clear list in-place
    history.clear()
    
    # Restore System Prompt
    if system_prompt_item:
        history.append(system_prompt_item)

    # Restore Trigger Message
    if trigger_message:
        history.append(trigger_message)
        
    status_msg.append("Conversation history cleared (system prompt and trigger preserved).")
    
    return " | ".join(status_msg)
