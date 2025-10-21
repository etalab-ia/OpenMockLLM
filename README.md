# OpenMockLLM

A FastAPI-based mock LLM API server that simulates multiple Large Language Model API providers. Currently supports **vLLM** (OpenAI-compatible) and **Mistral AI** endpoints.

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
# Using vLLM backend
python -m openmockllm.main --backend vllm --port 8000

# Using Mistral backend
python -m openmockllm.main --backend mistral --port 8001

# With custom configuration
python -m openmockllm.main \
  --backend vllm \
  --port 8000 \
  --max-context 128000 \
  --owned-by "MyOrganization" \
  --model-name "my-custom-model"
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--backend` | str | **required** | Backend to use: `vllm` or `mistral` |
| `--port` | int | `8000` | Port to run the server on |
| `--max-context` | int | `128000` | Maximum context length |
| `--owned-by` | str | `OpenMockLLM` | Owner of the API |
| `--model-name` | str | `openmockllm` | Model name to return in responses |
| `--embedding-dimension` | int | `1024` | Embedding dimension |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
