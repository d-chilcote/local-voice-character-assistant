import subprocess
import time
import os
import sys
import requests
import signal
import socket

# Configuration
TEST_PORT = 8900
SERVER_URL = f"http://localhost:{TEST_PORT}"
CHAT_URL = f"{SERVER_URL}/chat/text"
MODEL_PATH = os.path.expanduser("~/ai/models/Qwen3-8B-Q4_K_M.gguf")
SERVER_LOG_FILE = "server_test_log.txt"

def wait_for_server(process, max_retries=120):
    """Waits for the server to become responsive."""
    print(f"Waiting for server on {SERVER_URL}...")
    for i in range(max_retries):
        # Check if process died
        if process.poll() is not None:
            print("Server process died early.")
            return False
            
        try:
            resp = requests.get(f"{SERVER_URL}/v1/", timeout=2)
            if resp.status_code == 200:
                print("Server is up!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.RequestException:
            pass
            
        time.sleep(1)
    return False

def run_query(text):
    """Sends a text query to the server."""
    try:
        resp = requests.post(CHAT_URL, json={"text": text}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Query failed: {e}")
        return None

def test_skill(skill_name, prompt, expected_keywords):
    """Runs a test case for a specific skill."""
    print(f"\n[TEST] {skill_name}: '{prompt}'")
    result = run_query(prompt)
    
    if not result:
        print(f"❌ {skill_name} FAILED: No response")
        return False
        
    speech = result.get("speech", "")
    print(f"  -> Response: {speech}")
    
    # Check for keywords (case-insensitive)
    speech_lower = speech.lower()
    for kw in expected_keywords:
        if kw.lower() in speech_lower:
            print(f"✅ {skill_name} PASSED (Found '{kw}')")
            return True
            
    print(f"❌ {skill_name} FAILED: Missing keywords {expected_keywords}")
    return False

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Qwen model not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"🚀 Starting verification with model: {os.path.basename(MODEL_PATH)}")
    
    # 1. Start Server
    env = os.environ.copy()
    env["LLAMA_PATH"] = MODEL_PATH
    env["PORT"] = str(TEST_PORT)
    env["DEFAULT_CHARACTER"] = "R2-67" 
    
    # Open log file for non-blocking IO
    log_f = open(SERVER_LOG_FILE, "w")
    
    print(f"Redirecting server output to {SERVER_LOG_FILE}")
    
    # Start process
    server_process = subprocess.Popen(
        [sys.executable, "-u", "server_voice_assistant.py"],
        env=env,
        stdin=subprocess.DEVNULL, # Force non-interactive mode
        stdout=log_f,
        stderr=subprocess.STDOUT, 
        text=True
    )
    
    try:
        if not wait_for_server(server_process):
            print("Server failed to start.")
            
            # Check if it died or timed out
            if server_process.poll() is None:
                print("Terminating hanging server...")
                server_process.terminate()
            
            print(f"--- TAIL OF {SERVER_LOG_FILE} ---")
            log_f.flush()
            # Read last lines of log
            with open(SERVER_LOG_FILE, "r") as f:
                print(f.read()[-2000:]) # Print last 2000 chars
            print("---------------------")
            sys.exit(1)
            
        # 2. Run Tests
        results = []
        
        # System Info
        results.append(test_skill(
            "System Info", 
            "Get local system diagnostics", 
            ["OS", "CPU", "Memory", "macOS", "Apple"]
        ))
        
        # Calculator
        results.append(test_skill(
            "Calculator", 
            "Calculate 12345 times 2", 
            ["24690", "24,690"]
        ))
        
        # Google Search (Requires internet)
        results.append(test_skill(
            "Google Search", 
            "Who is the CEO of Google?", 
            ["Sundar Pichai"]
        ))
        
        # Todo List Add
        results.append(test_skill(
            "Todo List (Add)", 
            "Remind me to buy plutonium", 
            ["added", "list", "buy plutonium"]
        ))
        
        # Todo List Read
        results.append(test_skill(
            "Todo List (Read)", 
            "Read my todo list", 
            ["plutonium"]
        ))
        
        # Memory Erasure (Run last because it wipes context)
        results.append(test_skill(
            "Memory Erasure", 
            "You have amnesia", 
            ["erased", "cleared", "forgot", "memory"]
        ))
        
        # 3. Report
        print("\n" + "="*30)
        print("SUMMARY")
        print("="*30)
        if all(results):
            print("🎉 ALL SKILLS PASSED")
            sys.exit(0)
        else:
            print("💥 SOME SKILLS FAILED")
            sys.exit(1)
            
    finally:
        print("\nStopping server...")
        if server_process.poll() is None:
            server_process.terminate()
            server_process.wait()
        log_f.close()

if __name__ == "__main__":
    main()
