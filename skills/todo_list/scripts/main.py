import os
import json

TODO_FILE = "user_todo.txt"

def execute(action: str, item: str = "") -> str:
    """Manages todo list actions."""
    try:
        if action == "add":
            if not item: return "Error: No item provided to add."
            with open(TODO_FILE, "a") as f:
                f.write(f"- {item}\n")
            return f"Added: {item}"
            
        elif action == "read":
            if not os.path.exists(TODO_FILE):
                return "Todo list is empty."
            with open(TODO_FILE, "r") as f:
                content = f.read().strip()
            return content if content else "Todo list is empty."
            
        elif action == "clear":
            if os.path.exists(TODO_FILE):
                os.remove(TODO_FILE)
            return "Todo list cleared."
            
        else:
            return f"Unknown action: {action}"
            
    except Exception as e:
        return f"Todo Error: {e}"

if __name__ == "__main__":
    import sys
    # basic CLI test wrapper
    if len(sys.argv) > 1:
        print(execute(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""))
