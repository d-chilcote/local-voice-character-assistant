import os
import json
import pytest
from unittest.mock import MagicMock, patch
from core.agent_legacy import Agent
from skills.registry import SkillRegistry

# Mock LLM response to trigger the skill
MOCK_LLM_RESPONSE_TRIGGER = {
    "thought": "User wants me to forget everything.",
    "call_to_action": "memory_erasure",
    "speech": "Deleting my memory..."
}

MOCK_LLM_RESPONSE_AFTER = {
    "thought": "I have forgotten.",
    "call_to_action": "reply",
    "speech": "Who am I?"
}

class MockLLM:
    def create_chat_completion(self, **kwargs):
        # Return a mock response structure matching Agent expectation
        return {
            "choices": [{
                "message": {
                    "content": json.dumps(MOCK_LLM_RESPONSE_TRIGGER if "memory_erasure" not in str(kwargs) else MOCK_LLM_RESPONSE_AFTER)
                }
            }]
        }

@pytest.mark.asyncio
async def test_memory_erasure_skill(tmp_path):
    # 1. Setup
    memory_file = tmp_path / "test_memory.json"
    
    # Pre-populate memory
    history = [
        {"role": "system", "content": "You are a bot."},
        {"role": "user", "content": "My name is Dave."}
    ]
    with open(memory_file, "w") as f:
        json.dump(history, f)
        
    # Mock Registry to load our new skill (or just mock the execution)
    # Since we want to test the actual script, let's point registry to the real skills dir
    # BUT finding the real path in test is tricky. Let's just mock execute_skill 
    # OR better, stick to the plan: invoke the agent which invokes the registry.
    
    # Actually, let's trust the Registry finding the skill if we point it right?
    # No, simpler to verify the Agent logic invokes the skill with correct args including 'history' and 'memory_file'.
    
    mock_registry = MagicMock(spec=SkillRegistry)
    # We want to verify verify execute_skill receives 'history' and 'memory_file'
    
    llm = MockLLM()
    llm.create_chat_completion = MagicMock(side_effect=[
        {"choices": [{"message": {"content": json.dumps(MOCK_LLM_RESPONSE_TRIGGER)}}]}, # First call (Think)
        {"choices": [{"message": {"content": json.dumps(MOCK_LLM_RESPONSE_AFTER)}}]}    # Second call (After skill)
    ])
    
    agent = Agent(
        name="TestBot",
        system_prompt="SysPrompt",
        llm=llm,
        registry=mock_registry,
        config={"memory_file": str(memory_file)}
    )
    
    # 2. Execute
    # We pass a copy of history to the chat method, but the Agent *modifies* it (it appends).
    # But wait, the Agent usually loads history from file? No, it takes it as arg.
    in_memory_history = list(history)
    
    await agent.chat("Erase your memory", in_memory_history)
    
    # 3. Verify
    # Check that registry.execute_skill was called with correct args
    mock_registry.execute_skill.assert_called_once()
    args, kwargs = mock_registry.execute_skill.call_args
    assert args[0] == "memory_erasure"
    assert "history" in kwargs
    assert kwargs["history"] is in_memory_history # Should be the same object reference
    assert "memory_file" in kwargs
    assert kwargs["memory_file"] == str(memory_file)
    
    # Now let's verify the ACTUAL script logic (unit test the script directly)
    from skills.memory_erasure.scripts.erase import execute
    
    # Re-create file and history for script unit test
    with open(memory_file, "w") as f:
        json.dump(history, f)
    
    test_history = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Chat"}
    ]
    
    result = execute(test_history, str(memory_file))
    
    # Check File Deleted
    assert not os.path.exists(memory_file)
    
    # Check History Cleared (except system and trigger)
    assert len(test_history) == 2
    assert test_history[0]["role"] == "system"
    assert test_history[1]["role"] == "user"
    assert "cleared" in result

if __name__ == "__main__":
    pytest.main([__file__])
