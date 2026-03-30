import sys
import os
from unittest.mock import MagicMock

# Mock dependencies that are missing in the environment
mock_langgraph = MagicMock()
sys.modules["langgraph"] = mock_langgraph
sys.modules["langgraph.graph"] = mock_langgraph.graph
sys.modules["langgraph.graph.message"] = mock_langgraph.graph.message
sys.modules["langgraph.prebuilt"] = mock_langgraph.prebuilt

mock_langchain = MagicMock()
sys.modules["langchain_core"] = mock_langchain
sys.modules["langchain_core.messages"] = mock_langchain.messages

from core.agent_graph import parse_tool_call_from_text

def test_parsing():
    available_tools = ["google_search", "calculator"]

    # Test Qwen format
    content_qwen = "<tool_call>\n{\"name\": \"google_search\", \"arguments\": {\"query\": \"weather in London\"}}\n</tool_call>"
    result = parse_tool_call_from_text(content_qwen, available_tools)
    assert result == {"name": "google_search", "args": {"query": "weather in London"}}, f"Qwen parsing failed: {result}"

    # Test Nemotron format
    content_nemotron = "<function=google_search><parameter=query>weather in London</parameter></function>"
    result = parse_tool_call_from_text(content_nemotron, available_tools)
    assert result == {"name": "google_search", "args": {"query": "weather in London"}}, f"Nemotron parsing failed: {result}"

    # Test Heuristic - Weather
    content_weather = "I'll search for weather in London"
    result = parse_tool_call_from_text(content_weather, available_tools)
    assert result == {"name": "google_search", "args": {"query": "current weather in London"}}, f"Weather heuristic failed: {result}"

    # Test Heuristic - Calc
    content_calc = "we can use calculator to calculate 2 + 2"
    result = parse_tool_call_from_text(content_calc, available_tools)
    assert result == {"name": "calculator", "args": {"expression": "2 + 2"}}, f"Calc heuristic failed: {result}"

    print("All parsing tests passed!")

if __name__ == "__main__":
    try:
        test_parsing()
    except Exception as e:
        print(f"Tests failed: {e}")
        sys.exit(1)
