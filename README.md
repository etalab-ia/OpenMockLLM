# OpenMockLLM

A FastAPI-based mock LLM API server that simulates multiple Large Language Model API providers.

Supported backends:
| Backend | Description | Endpoints |
| --- | --- | --- |
| vLLM | OpenAI-compatible |• /v1/chat/completions<br>• /v1/models<br>• /v1/embeddings<br>• /health |
| Mistral | Mistral AI |• /v1/chat/completions<br>• /v1/models<br>• /v1/embeddings |
| TEI | Text Embeddings Inference |• /v1/embeddings<br>• /health<br>• /info<br>• /rerank |

## Quickstart

* Install the package:
  ```bash
  pip install git+https://github.com/etalab-ia/openmockllm.git
  ```

* Run the server:
  ```bash
  openmockllm
  ```

## Usage

### Command-line arguments

### Command-Line Arguments

#### Common Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--backend` | str | `vllm` | Backend to use: `vllm`, `mistral`, or `tei` |
| `--port` | int | `8000` | Port to run the server on |
| `--max-context` | int | `128000` | Maximum context length |
| `--owned-by` | str | `OpenMockLLM` | Owner of the API |
| `--model-name` | str | `openmockllm` | Model name to return in responses |
| `--embedding-dimension` | int | `1024` | Embedding dimension |
| `--api-key` | str | `None` | API key for authentication |
| `--tiktoken-encoder` | str | `cl100k_base` | Tiktoken encoder |
| `--faker-langage` | str | `fr_FR` | Langage used for generating prompt responses |
| `--faker-seed` | str | `None` | Seed for Faker generation |

#### TEI-Specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--payload-limit` | int | `2000000` | Payload size limit in bytes (2MB) |
| `--max-client-batch-size` | int | `32` | Maximum number of inputs per request |
| `--auto-truncate` | flag | `False` | Automatically truncate inputs longer than max size |
| `--max-batch-tokens` | int | `16384` | Maximum total tokens in a batch |

### Test Examples

#### Chat Completion (vLLM/Mistral)

* Streaming response:
```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
 -H "Content-Type: application/json" \
 -d '{ "model": "openmockllm", "messages": [{"role": "user", "content": "Bonjour"}], "stream": true }'
```

* Non-streaming response:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
 -H "Content-Type: application/json" \
 -d '{ "model": "openmockllm", "messages": [{"role": "user", "content": "Bonjour"}], "stream": false }'
```

#### Embeddings (TEI)

```bash
# Generate embeddings
curl -X POST http://localhost:8002/v1/embeddings \
 -H "Content-Type: application/json" \
 -d '{ "input": "Hello, world!", "model": "openmockllm" }'

# Get model info
curl http://localhost:8002/info

# Rerank documents
curl -X POST http://localhost:8002/rerank \
 -H "Content-Type: application/json" \
 -d '{ "query": "What is Deep Learning?", "texts": ["Deep Learning is...", "Machine Learning is..."] }'
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

```
# Install with pip
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
````

Run API with the following command:

```bash
python -m openmockllm.main --reload
```