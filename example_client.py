#!/usr/bin/env python3
"""
Example client for testing OpenMockLLM API

Make sure the server is running before executing this script:
    python -m openmockllm.main --backend vllm --port 8000
"""

import json

import requests


def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_models():
    """Test models listing endpoint"""
    print("=" * 60)
    print("Testing Models Endpoint")
    print("=" * 60)
    response = requests.get("http://localhost:8000/v1/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_chat_non_streaming():
    """Test non-streaming chat completion"""
    print("=" * 60)
    print("Testing Chat Completion (Non-streaming)")
    print("=" * 60)
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "openmockllm",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False,
        },
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_chat_streaming():
    """Test streaming chat completion"""
    print("=" * 60)
    print("Testing Chat Completion (Streaming)")
    print("=" * 60)
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "openmockllm",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": True,
        },
        stream=True,
    )
    print(f"Status: {response.status_code}")
    print("Stream chunks:")
    for i, line in enumerate(response.iter_lines()):
        if line:
            decoded = line.decode("utf-8")
            print(f"  Chunk {i}: {decoded}")
    print()


def test_embeddings():
    """Test embeddings endpoint"""
    print("=" * 60)
    print("Testing Embeddings Endpoint")
    print("=" * 60)
    response = requests.post(
        "http://localhost:8000/v1/embeddings",
        json={
            "model": "openmockllm",
            "input": "The quick brown fox jumps over the lazy dog",
        },
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    # Show a summary (full embedding vector would be too long)
    if "data" in result and len(result["data"]) > 0:
        embedding = result["data"][0]["embedding"]
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print(f"Usage: {result.get('usage', {})}")
    print()


def main():
    """Run all tests"""
    try:
        print("\n" + "=" * 60)
        print("OpenMockLLM API Client Test")
        print("=" * 60 + "\n")

        test_health()
        test_models()
        test_chat_non_streaming()
        test_chat_streaming()
        test_embeddings()

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the server.")
        print("Make sure the server is running:")
        print("    python -m openmockllm.main --backend vllm --port 8000\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
