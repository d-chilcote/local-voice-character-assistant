import os
import sys
from typing import Optional
from google import genai
from google.genai import types

# Add parent directory to path to import config and logger if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logger_config import get_logger
from config import cfg

logger = get_logger(__name__)

def execute_search(query: str, api_key: Optional[str] = None) -> str:
    """Performs a grounded Google Search via Gemini.
    
    Args:
        query: The search query string.
        api_key: Optional Gemini API key.
            
    Returns:
        The search results as a string.
    """
    api_key = api_key or cfg().gemini_api_key
    if not api_key:
        logger.error("No GEMINI_API_KEY found.")
        return "Search failed: No API key provided."

    try:
        client = genai.Client(api_key=api_key)
        model = "gemini-flash-lite-latest"
        
        response = client.models.generate_content(
            model=model,
            contents=f"Using Google Search, find the answer to this query: '{query}'.",
            config=types.GenerateContentConfig(
                max_output_tokens=1000, 
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
                tools=[types.Tool(google_search=types.GoogleSearch())],
                system_instruction="Provide a concise, factual answer.",
            )
        )
        
        return response.text
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Search failed: {e}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(execute_search(" ".join(sys.argv[1:])))
    else:
        print("Usage: python search.py <query>")
