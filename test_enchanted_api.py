#!/usr/bin/env python3
"""
Test harness to simulate Enchanted app API calls.

Usage:
    python test_enchanted_api.py "What's the weather in Utah?"
    python test_enchanted_api.py --interactive
"""
import argparse
import requests
import json
import sys

DEFAULT_HOST = "http://127.0.0.1:8888"


def send_chat_message(host: str, message: str, model: str = "r2_67") -> str:
    """Send a chat message using OpenAI-compatible API (like Enchanted does)."""
    url = f"{host}/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": message}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        print(f"\n📤 Sending to {url}...")
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")
            return content
        else:
            return f"Unexpected response: {data}"
    except requests.exceptions.ConnectionError:
        return "❌ Connection failed. Is the server running?"
    except requests.exceptions.Timeout:
        return "❌ Request timed out"
    except Exception as e:
        return f"❌ Error: {e}"


def send_legacy_api_chat(host: str, message: str) -> str:
    """Send using the legacy /v1/api/chat endpoint (POST with model in body)."""
    url = f"{host}/v1/api/chat"
    
    payload = {
        "model": "r2_67",
        "messages": [{"role": "user", "content": message}]
    }
    
    try:
        print(f"\n📤 Sending to {url}...")
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        return data.get("response", data.get("content", str(data)))
    except Exception as e:
        return f"❌ Error: {e}"


def interactive_mode(host: str):
    """Interactive chat mode."""
    print(f"\n🤖 Interactive mode - connected to {host}")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 Goodbye!")
                break
            
            response = send_chat_message(host, user_input)
            print(f"\n🤖 R2-67: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Test harness for Enchanted-style API calls")
    parser.add_argument("message", nargs="?", help="Message to send")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Server URL (default: {DEFAULT_HOST})")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")
    parser.add_argument("--legacy", "-l", action="store_true", help="Use legacy /v1/api/chat endpoint")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.host)
    elif args.message:
        if args.legacy:
            response = send_legacy_api_chat(args.host, args.message)
        else:
            response = send_chat_message(args.host, args.message)
        print(f"\n🤖 Response:\n{response}\n")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
