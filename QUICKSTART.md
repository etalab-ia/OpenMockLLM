# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -e .
```

## Start the Server

### vLLM Backend

```bash
python -m openmockllm.main --backend vllm --port 8000
```

### Mistral Backend

```bash
python -m openmockllm.main --backend mistral --port 8000
```

## Test the API

### Option 1: Use the Example Client

```bash
# In a new terminal (while server is running)
python example_client.py
```

### Option 2: Use curl

```bash
# Test health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openmockllm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openmockllm",
    "input": "Hello world"
  }'
```

### Option 3: Use Python OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="openmockllm",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

## Mock Response Content

Chat completions return dynamically generated lorem ipsum text (3 paragraphs) using the `lorem-text` library.

## API Documentation

Visit http://localhost:8000/docs to see the interactive API documentation.

