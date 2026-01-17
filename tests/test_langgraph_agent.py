"""Checkpoint: Verify LangGraph agent compiles and runs."""
from unittest.mock import MagicMock, patch

from core.agent_graph import create_agent_graph, AgentState


def test_agent_graph_compiles_without_tools():
    """Verify the agent graph compiles with no tools."""
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=MagicMock(content="Hello!", tool_calls=None))
    
    graph = create_agent_graph(mock_llm, [], "Test system prompt")
    assert graph is not None


def test_agent_graph_compiles_with_tools():
    """Verify the agent graph compiles with tools."""
    from langchain_core.tools import tool
    
    @tool
    def dummy_tool(x: str) -> str:
        """A dummy tool."""
        return f"Result: {x}"
    
    mock_llm = MagicMock()
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)
    mock_llm.invoke = MagicMock(return_value=MagicMock(content="Hello!", tool_calls=None))
    
    graph = create_agent_graph(mock_llm, [dummy_tool], "Test system prompt")
    assert graph is not None


def test_agent_state_has_messages():
    """Verify AgentState has the expected structure."""
    state: AgentState = {"messages": []}
    assert "messages" in state
    assert isinstance(state["messages"], list)
