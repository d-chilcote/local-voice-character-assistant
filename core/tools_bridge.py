"""Bridge existing skills to LangChain tools.

Wraps the SkillRegistry's skills as LangChain @tool decorated functions
for use with LangGraph agents.
"""
from typing import List, Any

from langchain_core.tools import tool, BaseTool

from skills.registry import registry
from logger_config import get_logger

logger = get_logger(__name__)


def get_all_tools(history: List = None, memory_file: str = None, memory_saver: Any = None) -> List[BaseTool]:
    """Get all available tools for the agent, injecting required context.
    
    Args:
        history: Reference to the conversation history list.
        memory_file: Path to the persistent memory file.
        memory_saver: Optional LangGraph MemorySaver to clear on erasure.
        
    Returns:
        List of LangChain tool instances.
    """
    
    @tool
    def google_search(query: str) -> str:
        """Search Google for current information, news, or facts.
        
        Args:
            query: The search query string.
        """
        result = registry.execute_skill("google_search", query=query)
        return result or "No results found."

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression.
        
        Args:
            expression: The math expression to evaluate (e.g., "2 + 2 * 3").
        """
        result = registry.execute_skill("calculator", expression=expression)
        return result or "Error evaluating expression."

    @tool
    def system_info() -> str:
        """Get system information about the host machine."""
        result = registry.execute_skill("system_info")
        return result or "Error getting system info."

    @tool
    def todo_list(action: str, item: str = "") -> str:
        """Manage a todo list.
        
        Args:
            action: The action to perform: "add", "remove", or "list".
            item: The item to add or remove (not needed for "list").
        """
        result = registry.execute_skill("todo_list", action=action, item=item)
        return result or "Error with todo list."

    @tool
    def erase_memory(confirmation: str) -> str:
        """Erase conversation memory. Requires explicit confirmation.
        
        Args:
            confirmation: Must be exactly 'CONFIRMED' to proceed.
        """
        # Inject server-specific context for memory erasure
        result = registry.execute_skill(
            "memory_erasure", 
            confirmation=confirmation,
            history=history,
            memory_file=memory_file
        )
        

        return result or "Error erasing memory."

    tools = [google_search, calculator, system_info, todo_list, erase_memory]
    logger.info(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")
    return tools
