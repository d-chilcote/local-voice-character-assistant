import math

def execute(expression: str) -> str:
    """Evaluates a mathematical expression safely."""
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max, "pow": pow})
    
    try:
        # Evaluate using only the allowed math functions
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Math Error: {e}"

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(execute(sys.argv[1]))
    else:
        print("Usage: python main.py <expression>")
