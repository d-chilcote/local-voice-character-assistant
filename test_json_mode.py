import json
from llama_cpp import Llama

# Config
MODEL_PATH = "./Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

print("--- JSON MODE POC (V2) ---")
print(f"Loading {MODEL_PATH}...")

try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1, # Metal Support
        n_ctx=2048,
        verbose=False
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Schema Definition
SYSTEM_PROMPT = """You are a robot. Output valid JSON only.
Schema:
{
  "thought": "Internal reasoning",
  "action": "reply",
  "content": "What to say"
}
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Hello! Who are you?"}
]

print("\nGenerating with response_format={'type': 'json_object'}...")

try:
    response = llm.create_chat_completion(
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=200,
        stop=["<|eot_id|>", "}", "\n\n"] # Stop after closing brace (NOTE: grammar usually handles this but safety first)
    )

    content = response['choices'][0]['message']['content']
    print(f"\nRAW OUTPUT:\n{content}\n")

    # If the model stops early (before closing brace), the grammar might ensure it's closed?
    # Let's see. Most JSON grammars force a complete object.
    
    # Simple cleaner
    if not content.strip().endswith("}"):
         content += "}"

    # Validate JSON
    data = json.loads(content)
    print("✅ Valid JSON Parsed!")
    print(f"Thought: {data.get('thought')}")
    print(f"Action: {data.get('action')}")
    print(f"Content: {data.get('content')}")

except Exception as e:
    print(f"❌ Failed: {e}")
