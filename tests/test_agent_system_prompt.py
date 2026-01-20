import unittest
from unittest.mock import MagicMock
from core.agent_legacy import Agent

class TestAgentSystemPrompt(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_registry = MagicMock()
        self.mock_registry.get_skill_instructions.return_value = "SKILL_INSTRUCTIONS"
        self.system_prompt = "SYSTEM_PROMPT"
        self.agent = Agent(
            name="TestAgent",
            system_prompt=self.system_prompt,
            llm=self.mock_llm,
            registry=self.mock_registry
        )

    def test_ensure_system_prompt_caching(self):
        # Initial state: cache is None
        self.assertIsNone(self.agent._cached_full_system_prompt)

        # First call
        history = []
        new_history = self.agent._ensure_system_prompt(history)

        expected_full_prompt = "SYSTEM_PROMPT\n\nSKILL_INSTRUCTIONS"
        self.assertEqual(new_history[0]["content"], expected_full_prompt)
        self.assertEqual(self.agent._cached_full_system_prompt, expected_full_prompt)

        # Verify get_skill_instructions was called once
        self.mock_registry.get_skill_instructions.assert_called_once()

        # Second call
        history_2 = []
        new_history_2 = self.agent._ensure_system_prompt(history_2)

        # Should be same content
        self.assertEqual(new_history_2[0]["content"], expected_full_prompt)

        # Verify get_skill_instructions was NOT called again (cached)
        self.mock_registry.get_skill_instructions.assert_called_once()

    def test_ensure_system_prompt_updates_history(self):
        # Case where history already has a system prompt
        full_prompt = "SYSTEM_PROMPT\n\nSKILL_INSTRUCTIONS"
        # Pre-populate cache to simulate steady state
        self.agent._cached_full_system_prompt = full_prompt

        history = [{"role": "system", "content": "OLD_PROMPT"}]
        new_history = self.agent._ensure_system_prompt(history)

        self.assertEqual(new_history[0]["content"], full_prompt)

    def test_ensure_system_prompt_no_update_if_same(self):
         # Case where history already has the correct system prompt
        full_prompt = "SYSTEM_PROMPT\n\nSKILL_INSTRUCTIONS"
        self.agent._cached_full_system_prompt = full_prompt

        history = [{"role": "system", "content": full_prompt}]
        # We want to check if the content object remains the same identity or if we can infer optimization
        # The code is:
        # if history[0]["content"] != full_system_prompt:
        #     history[0]["content"] = full_system_prompt

        # If we use a string subclass or mock, we might be able to detect access, but strings are immutable.
        # However, we can check if the assignment happened by using a setter property if we could...
        # but we are modifying a dict.

        new_history = self.agent._ensure_system_prompt(history)
        self.assertEqual(new_history[0]["content"], full_prompt)

if __name__ == '__main__':
    unittest.main()
