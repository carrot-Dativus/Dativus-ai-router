
import requests
import json

def test_stream():
    url = "http://127.0.0.1:8000/api/v1/chat/stream"
    payload = {
        "query": "안녕?",
        "workspace_id": "test_ws",
        "history": []
    }
    # Note: We need a token. Let's see if we can get one or bypass for test.
    # Since I don't have a valid token easily, I'll check how verify_token works.
    
    print(f"Testing stream to {url}...")
    try:
        # This will probably fail with 401 if token is required.
        response = requests.post(url, json=payload, stream=True)
        print(f"Response status: {response.status_code}")
        for line in response.iter_lines():
            if line:
                print(f"Received: {line.decode('utf-8')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_stream()
