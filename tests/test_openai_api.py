import requests
import json

BASE_URL = "http://localhost:8888/v1"

def test_list_models():
    print("\n--- Testing /v1/models ---")
    try:
        response = requests.get(f"{BASE_URL}/models")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200
        assert "data" in response.json()
        print("✅ Models list verified.")
    except Exception as e:
        print(f"❌ Models list failed: {e}")

def test_chat_completions():
    print("\n--- Testing /v1/chat/completions ---")
    payload = {
        "model": "r2_67",
        "messages": [
            {"role": "user", "content": "Hello, who are you?"}
        ]
    }
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", json=payload)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response Content: {data['choices'][0]['message']['content']}")
        assert response.status_code == 200
        assert "choices" in data
        print("✅ Chat completions verified.")
    except Exception as e:
        print(f"❌ Chat completions failed: {e}")

if __name__ == "__main__":
    # Note: Server must be running for this to work
    test_list_models()
    test_chat_completions()
