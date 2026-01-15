# OpenMockLLM

A FastAPI-based mock LLM API server that simulates multiple Large Language Model API providers.

Supported backends:
| Backend | Endpoints |
| --- | --- |
| [vLLM](https://github.com/vllm-project/vllm) |• /v1/chat/completions<br>• /v1/models<br>• /health |
| [Mistral](https://mistral.ai/) |• /v1/chat/completions<br>• /v1/models<br>• /v1/embeddings |
| [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) |• /v1/embeddings<br>• /health<br>• /info<br>• /rerank |

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
| `--simulate-latency` | flag | `False` | Simulate latency |
| `--reference-tps` | int | `100` | Reference tokens per second for latency simulation |

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

### Development

1. Install the dependencies:
```bash
pip install -e ".[dev]"
```

2. Run a server:
```bash
python -m openmockllm.main --reload --backend mistral
```

### Generate Pydantic schemas

* From openapi.json file:  
```
BACKEND=tei
datamodel-codegen --input docs/${BACKEND}  --input-file-type openapi --output openmockllm/${BACKEND}/schemas.py --output-model-type pydantic_v2.BaseModel --strict-nullable
```

Another recommand method is to use official SDK of the backend. 

### Testing

1. Install the dependencies:
```bash
pip install -e ".[dev]"
```

2. Run a server:
```bash
python -m openmockllm.main --reload --backend mistral
```

3. Run the tests:
```bash
pytest tests/test_mistral
```