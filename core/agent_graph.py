"""LangGraph ReAct agent with tool calling.

Provides a factory function to create a compiled LangGraph agent
that follows the Think → Act → Observe → Reply loop.
"""
import re
from datetime import datetime
from typing import TypedDict, Annotated, List, Optional, Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage

from logger_config import get_logger

logger = get_logger(__name__)


class AgentState(TypedDict):
    """Typed state for the LangGraph agent.
    
    Attributes:
        messages: Conversation history with automatic message accumulation.
    """
    messages: Annotated[List, add_messages]


def find_json_block(text: str) -> Optional[str]:
    """Finds the first complete JSON object in a string using brace counting."""
    start = text.find('{')
    if start == -1:
        return None
        
    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start:i+1]
    return None

def parse_tool_call_from_text(content: str, available_tools: List[str]) -> Optional[Dict[str, Any]]:
    """Manual parser for models that emit tool calls in text format."""
    if not content:
        return None
    
    # First, try to parse Qwen3's <tool_call> JSON format
    tool_call_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL)
    if tool_call_match:
        try:
            import json
            json_str = tool_call_match.group(1).strip()
            # Use our robust finder for the inner content
            block = find_json_block(json_str)
            if block:
                tool_data = json.loads(block)
                tool_name = tool_data.get("name") or tool_data.get("function")
                tool_args = tool_data.get("arguments") or tool_data.get("args") or tool_data.get("parameters") or {}
                if tool_name and tool_name in available_tools:
                    logger.info(f"[PARSER] Extracted Qwen3 tool_call: {tool_name}({tool_args})")
                    return {"name": tool_name, "args": tool_args}
        except Exception as e:
            logger.warning(f"[PARSER] Failed to parse Qwen3 tool_call JSON: {e}")
    
    # Second, try Nemotron's <function=X><parameter=Y> XML format
    # Format: <function=google_search>\n<parameter=query>\nvalue\n</parameter>\n</function>
    func_match = re.search(r'<function=(\w+)>(.*?)</function>', content, re.DOTALL)
    if func_match:
        tool_name = func_match.group(1)
        func_body = func_match.group(2)
        if tool_name in available_tools:
            # Extract parameters
            tool_args = {}
            param_matches = re.findall(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', func_body, re.DOTALL)
            for param_name, param_value in param_matches:
                tool_args[param_name] = param_value.strip()
            logger.info(f"[PARSER] Extracted Nemotron function: {tool_name}({tool_args})")
            return {"name": tool_name, "args": tool_args}
    
    # Third, try raw JSON anywhere in the text (for Llama-3.1 behavior)
    json_block = find_json_block(content)
    if json_block:
        try:
            import json
            tool_data = json.loads(json_block)
            tool_name = tool_data.get("name") or tool_data.get("function")
            tool_args = tool_data.get("arguments") or tool_data.get("args") or tool_data.get("parameters") or {}
            if tool_name and tool_name in available_tools:
                logger.info(f"[PARSER] Extracted embedded JSON tool_call: {tool_name}({tool_args})")
                return {"name": tool_name, "args": tool_args}
        except Exception:
            pass
    
    
    # Fallback: ONLY use heuristic on first response (before any tool results)
    # Skip if this content looks like it already has tool result context
    content_lower = content.lower()
    if "search results" in content_lower or "tool result" in content_lower or "the search" in content_lower:
        logger.debug("[PARSER] Skipping heuristic - appears to be post-tool response")
        return None
    
    # Heuristic for Nemotron/Devstral: "We can use google_search tool" pattern
    # Only trigger if model explicitly mentions using a specific tool
    for tool_name in available_tools:
        use_patterns = [
            f"we can use {tool_name}",
            f"use {tool_name} tool",
            f"use the {tool_name}",
            f"call {tool_name}",
            f"using {tool_name}",
            f"i need to search",
            f"i should search",
            f"let me search",
            f"i'll search",
            f"search for",
        ]
        if any(pattern in content_lower for pattern in use_patterns):
            logger.info(f"[PARSER] Heuristic matched: model wants to use {tool_name}")
            # Extract query context from the user's question
            if tool_name == "google_search":
                # Try to find weather location or other search context
                weather_match = re.search(r"weather\s+(?:in\s+)?([a-zA-Z\s]+?)(?:\s+today|\"|\'|\.|,|$)", content, re.IGNORECASE)
                if weather_match:
                    query = f"current weather in {weather_match.group(1).strip()}"
                    logger.info(f"[PARSER] Extracted query: {query}")
                    return {"name": "google_search", "args": {"query": query}}
                # Try to find specific questions in quotes
                user_q_match = re.search(r'user asks?\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
                if user_q_match:
                    return {"name": "google_search", "args": {"query": user_q_match.group(1)}}
                # Try extracting from "about X" or "for X" patterns
                about_match = re.search(r'(?:about|for)\s+["\']?([^"\'\.]+?)["\']?(?:\.|,|$)', content, re.IGNORECASE)
                if about_match:
                    query = about_match.group(1).strip()
                    if len(query) > 5:  # Avoid tiny matches
                        logger.info(f"[PARSER] Extracted 'about' query: {query}")
                        return {"name": "google_search", "args": {"query": query}}
            elif tool_name == "calculator":
                expr_match = re.search(r'calculate\s+([0-9+\-*/\s().]+)', content, re.IGNORECASE)
                if expr_match:
                    return {"name": "calculator", "args": {"expression": expr_match.group(1).strip()}}
            break  # Only match one tool
    
    return None


def create_agent_graph(
    llm,
    tools: List,
    system_prompt: str,
    checkpointer=None
):
    """Build a LangGraph ReAct agent."""
    tool_names = [t.name for t in tools] if tools else []
    tools_dict = {t.name: t for t in tools} if tools else {}
    
    if tools:
        logger.info(f"Binding {len(tools)} tools to LLM: {tool_names}")
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm
    
    async def call_model(state: AgentState):
        """Node: Call the LLM with current messages."""
        msgs = state["messages"]
        logger.info(f"[AGENT] Incoming messages count: {len(msgs)}")
        
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        full_prompt = f"{system_prompt}\n\nCurrent Date: {current_date}"
        
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=full_prompt)] + list(msgs)
        else:
            # Refresh personality and date
            msgs[0] = SystemMessage(content=full_prompt)
        
        response = await llm_with_tools.ainvoke(msgs)
        
        # Log response details
        logger.info(f"[AGENT] Response type: {type(response).__name__}")
        content_preview = str(response.content)[:500] if response.content else "(empty)"
        logger.info(f"[AGENT] Response content: {content_preview}...")
        
        native_calls = getattr(response, 'tool_calls', []) or []
        logger.info(f"[AGENT] Native tool_calls: {native_calls}")
        
        # Fallback: parse tool calls from content if native is empty
        if not native_calls and response.content and tools:
            parsed = parse_tool_call_from_text(response.content, tool_names)
            if parsed:
                logger.info(f"[AGENT] Fallback parser found tool call: {parsed}")
                # Execute the tool directly and create a new response
                tool_func = tools_dict.get(parsed["name"])
                if tool_func:
                    try:
                        tool_result = await tool_func.ainvoke(parsed["args"])
                        logger.info(f"[AGENT] Tool result: {str(tool_result)[:500]}...")
                        # Create a follow-up message with the result
                        return {
                            "messages": [
                                response,
                                ToolMessage(content=str(tool_result), tool_call_id="fallback"),
                            ]
                        }
                    except Exception as e:
                        logger.error(f"[AGENT] Fallback tool error: {e}")
        
        return {"messages": [response]}
    
    def should_continue(state: AgentState) -> str:
        """Edge: Decide whether to call tools or end."""
        if not state["messages"]:
            return END
            
        last_message = state["messages"][-1]
        
        # If last message is a ToolMessage, we need to call agent again
        if isinstance(last_message, ToolMessage):
            logger.info("[EDGE] ToolMessage detected, continuing to agent")
            return "agent"
        
        # Check for native tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info(f"[EDGE] Native tool calls: {[tc.get('name') for tc in last_message.tool_calls]}")
            return "tools"
        
        logger.info("[EDGE] No tool calls, ending")
        return END
    
    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    if tools:
        graph.add_node("tools", ToolNode(tools))
    
    graph.set_entry_point("agent")
    
    if tools:
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "agent": "agent", END: END})
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", END)
    
    return graph.compile(checkpointer=checkpointer)


