import asyncio
from datetime import datetime
import json
from typing import List, Dict, Any, Optional, Tuple
from llama_cpp import Llama
from logger_config import get_logger
from skills.registry import SkillRegistry

logger = get_logger(__name__)

class Agent:
    """
    A persistent agent that maintains identity and orchestrates the
    Think -> Act -> Observe -> Reply loop.
    """
    def __init__(
        self, 
        name: str, 
        system_prompt: str, 
        llm: Llama, 
        registry: SkillRegistry,
        config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.registry = registry
        self.config = config or {}

    async def chat(self, user_text: str, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
        """
        Processes a user message and returns the assistant's verbal response
        and the updated history.
        
        Args:
            user_text: The spoken input from the user.
            history: The conversation history (including system prompt).
            
        Returns:
            (response_speech, updated_history)
        """
        # 1. Update History with User Input
        # Note: Caller handles loading history; we just append to the list provided
        # But we must ensure the system prompt is up to date with skills
        history = self._ensure_system_prompt(history)
        
        history.append({"role": "user", "content": user_text})
        
        # 2. Think (Step 1: Plan/Decide)
        response_1 = await self._llm_json_completion(history)
        
        thought = response_1.get("thought", "No thought provided.")
        action = response_1.get("call_to_action", "reply")
        speech = response_1.get("speech", "...")
        
        logger.info(f"{self.name} (Thought): {thought}")
        
        final_response = speech

        # 3. Act (Skill Loop)
        if action and action not in ["reply", "none"]:
             final_response = await self._handle_action(action, response_1, history)
        else:
             logger.info(f"{self.name}: {final_response}")

        # 4. Finalize History
        history.append({"role": "assistant", "content": final_response})
        
        return final_response, history

    def _ensure_system_prompt(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Ensures the system prompt contains the latest skills."""
        skill_info = self.registry.get_skill_instructions()
        full_system_prompt = f"{self.system_prompt}\n\n{skill_info}" if skill_info else self.system_prompt
        
        if not history:
            return [{"role": "system", "content": full_system_prompt}]
        
        # Update existing system prompt if present
        if history[0]["role"] == "system":
            history[0]["content"] = full_system_prompt
        else:
            # Insert if missing (rare case)
            history.insert(0, {"role": "system", "content": full_system_prompt})
            
        return history

    async def _handle_action(self, action: str, ai_data: Dict[str, Any], history: List[Dict[str, str]]) -> str:
        """Executes a skill and performs the follow-up generation."""
        # Normalize action name (handle 'search' alias)
        target_action = "google_search" if action in ["search", "google_search"] else action
        
        # Extract Arguments
        skill_args = {}
        if target_action == "google_search":
             skill_args["query"] = ai_data.get("search_query") or ai_data.get("query")
        elif target_action == "calculator":
             skill_args["expression"] = ai_data.get("expression") or ai_data.get("search_query") or ai_data.get("query")
        else:
            # Generic mapping
            for k, v in ai_data.items():
                if k not in ["thought", "call_to_action", "speech"]:
                    skill_args[k] = v
                    
        logger.info(f"{self.name} (Action): Executing {target_action} with {skill_args}...")
        
        # Execute
        skill_result = self.registry.execute_skill(target_action, **skill_args)
        
        if not skill_result:
            return "I tried to use a skill, but it failed. Sorry."

        # Inject Result & Re-think
        current_date = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
        
        temp_history = list(history) # Copy to avoid mutating the main loop prematurely
        # We need to find the specific user message we are responding to
        # In this flow, it's the last one we just appended
        if temp_history and temp_history[-1]['role'] == 'user':
            last_msg = temp_history.pop()
            new_content = (
                f"{last_msg['content']}\n\n"
                f"[SYSTEM: {target_action.upper()} RESULTS]\n"
                f"DATE: {current_date}\n"
                f"DATA:\n{skill_result}\n\n"
                f"[INSTRUCTION]: Answer based on the data above. Output JSON with action='reply'."
            )
            temp_history.append({"role": "user", "content": new_content})
            
        response_2 = await self._llm_json_completion(temp_history)
        final_speech = response_2.get("speech", "Error generating speech.")
        
        logger.info(f"{self.name} (With Skill Result): {final_speech}")
        return final_speech

    async def _llm_json_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Wrapper for LLM JSON generation."""
        try:
            response = await asyncio.to_thread(
                self.llm.create_chat_completion,
                messages=messages,
                temperature=0.7,
                max_tokens=300,
                response_format={"type": "json_object"},
                stop=["}"]
            )
            raw = response['choices'][0]['message']['content']
            if not raw.strip().endswith("}"): raw += "}"
            return json.loads(raw)
        except Exception as e:
            logger.error(f"LLM/JSON Error: {e}")
            return {"thought": "Error", "call_to_action": "reply", "speech": "My brain hurts."}
