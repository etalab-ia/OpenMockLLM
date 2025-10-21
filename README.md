# OpenMockLLM

A FastAPI-based mock LLM API server that simulates multiple Large Language Model API providers.

Supported backends:
| Backend | Description | Endpoints |
| --- | --- | --- |
| vLLM | OpenAI-compatible |• /v1/chat/completions<br>• /v1/models<br>• /v1/embeddings<br>• /health |
| Mistral | Mistral AI |• /v1/chat/completions<br>• /v1/models<br>• /v1/embeddings |

## Installation

```bash
# Install with pip
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Usage

### Starting the Server

Run the server with the desired backend:

```bash
# Using vllm backend
python -m openmockllm.main --backend vllm --port 8000

# Using mistral backend
python -m openmockllm.main --backend mistral --port 8001

# With custom configuration
python -m openmockllm.main \
  --backend vllm \
  --port 8000 \
  --max-context 128000 \
  --owned-by "MyOrganization" \
  --model-name "my-custom-model"
```

### Test chat

```
curl -N -X POST http://localhost:8000/v1/chat/completions \
 -H "Content-Type: application/json" \
 -d '{ "model": "openmockllm", "messages": [{"role": "user", "content": "Bonjour"}], "stream": true }'
```


### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--backend` | str | `vllm` | Backend to use: `vllm` or `mistral` |
| `--port` | int | `8000` | Port to run the server on |
| `--max-context` | int | `128000` | Maximum context length |
| `--owned-by` | str | `OpenMockLLM` | Owner of the API |
| `--model-name` | str | `openmockllm` | Model name to return in responses |
| `--embedding-dimension` | int | `1024` | Embedding dimension |
| `--api-key` | str | `None` | API key for authentication |
| `--tiktoken-encoder` | str | `cl100k_base` | Tiktoken encoder |
| `--faker-langage` | str | `fr_FR` | Langage used for generating prompt responses |
| `--faker-seed-instance` | str | `None` | Seed for Faker generation |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
