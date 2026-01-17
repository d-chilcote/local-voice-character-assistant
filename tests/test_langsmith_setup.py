"""Checkpoint: Verify LangSmith configuration (optional)."""
import os


def test_langsmith_env_vars_optional():
    """
    LangSmith is optional. If configured, verify the env vars are correct.
    If not configured, test passes (tracing is simply disabled).
    """
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    
    if tracing_enabled:
        api_key = os.getenv("LANGCHAIN_API_KEY")
        assert api_key, "LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is missing"
        assert api_key != "your-api-key-here", "Replace placeholder API key"
        print(f"✓ LangSmith tracing enabled, project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    else:
        print("✓ LangSmith tracing disabled (optional)")


def test_langchain_imports():
    """Verify LangChain packages are installed correctly."""
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langgraph.graph import StateGraph
    
    assert HumanMessage is not None
    assert StateGraph is not None
    print("✓ LangChain/LangGraph imports successful")
