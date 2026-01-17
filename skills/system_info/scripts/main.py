import platform
import psutil
import json

def execute() -> str:
    """Returns system diagnostics."""
    try:
        info = {
            "os": f"{platform.system()} {platform.release()}",
            "architecture": platform.machine(),
            "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "python_version": platform.python_version()
        }
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Error retrieving system info: {e}"

if __name__ == "__main__":
    print(execute())
