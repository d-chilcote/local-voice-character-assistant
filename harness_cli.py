import requests
import json
import sys

def run_harness():
    server_url = "http://127.0.0.1:8000/chat/text"
    
    print("\n" + "="*50)
    print("🤖 AGENT TEXT HARNESS (v1.0)")
    print("Type your message and press Enter. Type 'exit' to quit.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                break
            
            if not user_input:
                continue

            response = requests.post(
                server_url,
                json={"text": user_input},
                timeout=60 # Agent reasoning can take time
            )
            
            if response.status_code == 200:
                data = response.json()
                speech = data.get("speech", "No response.")
                print(f"\nAssistant: {speech}\n")
            else:
                print(f"\n[ERROR] Server returned {response.status_code}: {response.text}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[ERROR] Failed to connect to agent: {e}")
            break

if __name__ == "__main__":
    run_harness()
