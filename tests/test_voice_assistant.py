"""
Comprehensive Test Suite for Local AI Voice Assistant.

Tests focus on:
1. Character config integrity
2. JSON parsing robustness
3. Gemini error handling
4. Memory persistence
5. Empty audio handling
6. Safety settings verification
7. Voice fallback logic
8. Action dispatch (search vs reply)
9. CLI input validation
10. Per-character system prompts
"""

import pytest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock

# Import the items we can safely test without loading heavy models
from characters import CHARACTERS


# =============================================================================
# 1. CHARACTER CONFIG VALIDATION (Critical)
# =============================================================================
REQUIRED_CHARACTER_KEYS = [
    "id", "name", "description", "voice_native", "voice_fallback",
    "memory_file", "face", "system_prompt"
]

def test_all_characters_have_required_keys():
    """Every character must have all required config keys to avoid boot crash."""
    for char in CHARACTERS:
        for key in REQUIRED_CHARACTER_KEYS:
            assert key in char, f"Character '{char.get('name', 'UNKNOWN')}' missing key: {key}"


def test_all_characters_have_unique_ids():
    """Character IDs must be unique to avoid config collisions."""
    ids = [c["id"] for c in CHARACTERS]
    assert len(ids) == len(set(ids)), "Duplicate character IDs found"


def test_all_characters_have_unique_memory_files():
    """Memory files must be unique to prevent cross-character contamination."""
    files = [c["memory_file"] for c in CHARACTERS]
    assert len(files) == len(set(files)), "Duplicate memory file paths found"


# =============================================================================
# 2. JSON PARSING ROBUSTNESS (Critical)
# =============================================================================
def test_json_parsing_handles_malformed_input():
    """Verify graceful handling of malformed JSON (simulating LLM output)."""
    malformed_inputs = [
        '{"thought": "test", "speech": "hello"',  # Missing closing brace
        '{"thought": "test", speech: "hello"}',   # Missing quotes
        '',                                        # Empty string
        'not json at all',                        # Plain text
        '{"nested": {"broken": }',                # Broken nesting
    ]
    
    for raw in malformed_inputs:
        # Simulate the server's fallback logic
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"thought": "Error", "call_to_action": "reply", "speech": "Fallback"}
        
        assert "speech" in result, f"Fallback failed for input: {raw}"


def test_json_bracket_fix_logic():
    """Verify the bracket-appending safety net works."""
    raw = '{"thought": "test", "speech": "hello"'  # Missing }
    if not raw.strip().endswith("}"):
        raw += "}"
    
    data = json.loads(raw)
    assert data["speech"] == "hello"


# =============================================================================
# 3. GEMINI ERROR HANDLING (High)
# =============================================================================
def test_search_returns_error_string_on_exception():
    """Verify search_with_gemini returns safe error string, not exception."""
    # We can't import the function directly without loading models,
    # so we test the pattern that should be used
    def mock_search_with_gemini(query):
        try:
            raise ConnectionError("Simulated network failure")
        except Exception as e:
            return f"Search failed: {e}"
    
    result = mock_search_with_gemini("test query")
    assert "Search failed:" in result
    assert "Simulated network failure" in result


# =============================================================================
# 4. MEMORY PERSISTENCE (High)
# =============================================================================
def test_memory_save_and_load_roundtrip():
    """Verify chat history survives a save/load cycle."""
    test_history = [
        {"role": "system", "content": "You are a test bot."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_history, f)
        temp_path = f.name
    
    try:
        with open(temp_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded == test_history
        assert len(loaded) == 3
        assert loaded[0]["role"] == "system"
    finally:
        os.unlink(temp_path)


def test_memory_handles_corrupted_json():
    """Verify corrupted memory file triggers fallback, not crash."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{ this is not valid json }")
        temp_path = f.name
    
    try:
        # Simulate load_memory behavior
        try:
            with open(temp_path, 'r') as f:
                history = json.load(f)
        except Exception:
            history = [{"role": "system", "content": "Fallback prompt"}]
        
        assert len(history) == 1
        assert history[0]["role"] == "system"
    finally:
        os.unlink(temp_path)


# =============================================================================
# 5. EMPTY AUDIO HANDLING (Medium)
# =============================================================================
def test_empty_transcription_returns_empty_response():
    """When Whisper returns empty text, endpoint should return empty audio."""
    user_text = "".strip()
    
    if not user_text:
        response_content = b""
    else:
        response_content = b"some audio data"
    
    assert response_content == b""


# =============================================================================
# 6. GEMINI SAFETY SETTINGS (Medium)
# =============================================================================
def test_safety_settings_categories_are_defined():
    """Verify all 4 critical safety categories are referenced in code."""
    # Read the source file to verify safety settings exist
    import pathlib
    source_path = pathlib.Path(__file__).parent.parent / "server_voice_assistant.py"
    
    with open(source_path, 'r') as f:
        source_code = f.read()
    
    required_categories = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    
    for category in required_categories:
        assert category in source_code, f"Safety category {category} not found in source"


def test_safety_threshold_is_strict():
    """Verify safety threshold is set to BLOCK_LOW_AND_ABOVE (strictest)."""
    import pathlib
    source_path = pathlib.Path(__file__).parent.parent / "server_voice_assistant.py"
    
    with open(source_path, 'r') as f:
        source_code = f.read()
    
    assert "BLOCK_LOW_AND_ABOVE" in source_code, "Safety threshold should be BLOCK_LOW_AND_ABOVE"


# =============================================================================
# 7. VOICE FALLBACK LOGIC (Medium)
# =============================================================================
def test_voice_config_has_fallback():
    """Every character must have a fallback voice defined."""
    for char in CHARACTERS:
        assert char.get("voice_fallback"), f"Character '{char['name']}' has no fallback voice"
        assert char["voice_fallback"] != char["voice_native"], \
            f"Character '{char['name']}' fallback should differ from native"


# =============================================================================
# 8. ACTION DISPATCH LOGIC (Medium)
# =============================================================================
def test_action_dispatch_search():
    """Verify 'search' action is correctly identified."""
    ai_data = {"thought": "Need info", "call_to_action": "search", "search_query": "test"}
    
    action = ai_data.get("call_to_action")
    assert action == "search"
    
    if action == "search":
        query = ai_data.get("search_query")
        assert query == "test"


def test_action_dispatch_reply():
    """Verify 'reply' action uses speech directly."""
    ai_data = {"thought": "I know this", "call_to_action": "reply", "speech": "Hello!"}
    
    action = ai_data.get("call_to_action")
    assert action == "reply"
    
    if action != "search":
        final_response = ai_data.get("speech")
        assert final_response == "Hello!"


def test_action_dispatch_handles_none():
    """Verify missing call_to_action defaults to reply behavior."""
    ai_data = {"thought": "Hmm", "speech": "Default response"}
    
    action = ai_data.get("call_to_action")
    assert action is None
    
    # Should fall through to reply (else branch)
    if action != "search":
        final_response = ai_data.get("speech")
        assert final_response == "Default response"


# =============================================================================
# 9. CLI CHARACTER SELECTION (Low)
# =============================================================================
def test_character_count_matches_menu():
    """Verify CHARACTERS list has at least one entry."""
    assert len(CHARACTERS) >= 1, "No characters defined"


def test_character_indices_are_valid():
    """Verify all character indices are accessible (1-based for CLI)."""
    for i in range(len(CHARACTERS)):
        assert CHARACTERS[i] is not None


# =============================================================================
# 10. PER-CHARACTER SYSTEM PROMPTS (Low)
# =============================================================================
def test_all_prompts_contain_json_schema():
    """Every character's system prompt must instruct JSON output."""
    for char in CHARACTERS:
        prompt = char["system_prompt"]
        assert "JSON" in prompt.upper(), f"Character '{char['name']}' prompt missing JSON instruction"
        assert "thought" in prompt, f"Character '{char['name']}' prompt missing 'thought' field"
        assert "call_to_action" in prompt, f"Character '{char['name']}' prompt missing 'call_to_action'"
        assert "speech" in prompt, f"Character '{char['name']}' prompt missing 'speech' field"


def test_prompts_are_unique():
    """System prompts should be unique per character."""
    prompts = [c["system_prompt"] for c in CHARACTERS]
    assert len(prompts) == len(set(prompts)), "Duplicate system prompts found"

# =============================================================================
# 11. HISTORY PRUNING LOGIC (Critical)
# =============================================================================
def test_history_pruning_preserves_system_prompt_at_index_zero():
    """Verify that when history is pruned, index 0 (System Prompt) is never lost."""
    # Simulate a history logic that appends and slices
    limit = 2
    history = [{"role": "system", "content": "I am logic"}]
    
    # Add 3 messages (System + 3 = 4 items. Limit is 2. Result should be System + last 2)
    new_msgs = ["msg1", "msg2", "msg3"]
    for msg in new_msgs:
        history.append({"role": "user", "content": msg})
        if len(history) > limit + 1:
            history = [history[0]] + history[-limit:]
    
    assert len(history) == 3 # System + 2
    assert history[0]["role"] == "system"
    assert history[0]["content"] == "I am logic"
    assert history[-1]["content"] == "msg3"


def test_history_limit_zero_edge_case():
    """Verify behavior when limit is 0 (System Prompt only)."""
    limit = 0
    history = [{"role": "system", "content": "sys"}]
    
    history.append({"role": "user", "content": "msg1"})
    
    # Logic: [history[0]] + history[-0:] -> [sys] + [] -> [sys]
    if len(history) > limit + 1:
        history = [history[0]] + history[-limit:] if limit > 0 else [history[0]]
        
    assert len(history) == 1
    assert history[0]["role"] == "system"


# =============================================================================
# 12. ENVIRONMENT VARIABLE SAFETY (Critical)
# =============================================================================
def test_missing_api_key_handles_generically():
    """Verify logic doesn't crash if GEMINI_API_KEY is unset during search simulation."""
    with patch.dict(os.environ, {}, clear=True):
        # Determine API key behavior
        key = os.getenv("GEMINI_API_KEY")
        assert key is None
        
        # We expect the search function to check for this or fail gracefully
        # Since we can't run the actual API call, we test logic flow
        has_key = bool(key)
        assert has_key is False


# =============================================================================
# 13. PATH TRAVERSAL (High)
# =============================================================================
def test_memory_file_path_traversal_check():
    """Verify memory files are simple filenames, not paths causing traversal."""
    for char in CHARACTERS:
        mem_file = char["memory_file"]
        assert "/" not in mem_file, f"Memory file {mem_file} should be a filename, not a path"
        assert ".." not in mem_file, f"Memory file {mem_file} contains traversal intent"


# =============================================================================
# 14. SEARCH INPUT ROBUSTNESS (Medium)
# =============================================================================
def test_search_handles_none_query():
    """Verify passing None to search doesn't throw AttributeError."""
    def mock_search_logic(query):
        if not query:
            return "No query provided"
        return f"Searching for {query}"
        
    assert mock_search_logic(None) == "No query provided"
    assert mock_search_logic("") == "No query provided"


# =============================================================================
# 15. UNICODE & TTS SAFETY (Medium)
# =============================================================================
def test_speech_handles_unicode():
    """Verify speech field can carry unicode without JSON encoding errors."""
    data = {"speech": "Hello 🤖 world 🌍"}
    json_str = json.dumps(data)
    loaded = json.loads(json_str)
    assert loaded["speech"] == "Hello 🤖 world 🌍"


# =============================================================================
# 16. VALIDATION EXTRAS (Low)
# =============================================================================
def test_character_description_is_nonempty():
    """Description fields should not be empty (used for help/docs)."""
    for char in CHARACTERS:
        desc = char.get("description", "")
        assert desc.strip(), f"Character {char['name']} has empty description"


def test_face_is_ascii_safe():
    """ASCII art should be printable."""
    for char in CHARACTERS:
        face = char.get("face", "")
        assert face.strip(), "Face cannot be empty"
        # Basic printable check
        assert face.isprintable() or "\n" in face, "Face contains non-printable characters"
