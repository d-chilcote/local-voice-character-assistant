"""Checkpoint: Verify tools bridge works."""
from core.tools_bridge import get_all_tools


def test_all_tools_have_names():
    """Verify all tools are properly defined with names."""
    tools = get_all_tools()
    assert len(tools) == 5, f"Expected 5 tools, got {len(tools)}"
    
    names = [t.name for t in tools]
    assert "google_search" in names
    assert "calculator" in names
    assert "system_info" in names
    assert "todo_list" in names
    assert "erase_memory" in names


def test_tools_have_descriptions():
    """Verify all tools have descriptions for the LLM."""
    tools = get_all_tools()
    for tool in tools:
        assert tool.description, f"Tool {tool.name} missing description"
        assert len(tool.description) > 10, f"Tool {tool.name} description too short"


def test_calculator_tool_execution():
    """Verify calculator tool can execute."""
    from core.tools_bridge import calculator
    result = calculator.invoke({"expression": "2 + 2"})
    assert "4" in str(result), f"Calculator returned unexpected: {result}"
