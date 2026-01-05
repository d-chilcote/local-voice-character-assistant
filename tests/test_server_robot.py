import pytest
import json
from server_robot import clean_response_text, update_chat_history, SYSTEM_PROMPT

# 1. System Prompt Schema Validation
def test_system_prompt_has_json_instruction():
    """Ensure the system prompt instructions for JSON are present."""
    assert "JSON ONLY" in SYSTEM_PROMPT
    assert '"thought"' in SYSTEM_PROMPT
    assert '"call_to_action"' in SYSTEM_PROMPT
    assert '"search_query"' in SYSTEM_PROMPT
    assert '"speech"' in SYSTEM_PROMPT

# 2. clean_response_text (Still useful for cleanup)
def test_clean_response_text():
    """Verify that tags are stripped from output text."""
    assert clean_response_text("Hello [JSON: ...]") == "Hello " 
    assert clean_response_text("Normal text") == "Normal text"

# 3. update_chat_history
def test_update_chat_history():
    """Verify sliding window logic for chat history."""
    history = [{"role": "system", "content": "sys"}]
    limit = 2
    
    # Add 1
    history = update_chat_history(history, "user", "msg1", limit)
    assert len(history) == 2
    assert history[1]["content"] == "msg1"
    
    # Add 2 (Total 3: sys, msg1, msg2)
    history = update_chat_history(history, "ai", "msg2", limit)
    assert len(history) == 3
    
    # Add 3 (Total 4 -> Prune to 3: sys, msg2, msg3)
    history = update_chat_history(history, "user", "msg3", limit)
    assert len(history) == 3 
    assert history[0]["content"] == "sys"
    assert history[1]["content"] == "msg2"
    assert history[2]["content"] == "msg3"
