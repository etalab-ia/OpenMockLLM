import argparse

from fastapi import FastAPI
import uvicorn

from openmockllm.logger import init_logger
from openmockllm.settings import settings

logger = init_logger("openmockllm")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="OpenMockLLM - Mock LLM API Server")
    parser.add_argument("--backend", type=str, choices=["vllm", "mistral", "tei"], default="vllm", help="Backend to use (vllm, mistral, or tei)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    parser.add_argument("--max-context", type=int, default=128000, help="Maximum context length (default: 128000)")
    parser.add_argument("--owned-by", type=str, default="OpenMockLLM", help="Owner of the API (default: OpenMockLLM)")
    parser.add_argument("--model-name", type=str, default="openmockllm", help="Model name to return (default: openmockllm)")
    parser.add_argument("--embedding-dimension", type=int, default=1024, help="Embedding dimension (default: 1024)")
    parser.add_argument("--api-key", type=str, default=None, help="API key for authentication (optional)")
    parser.add_argument("--tiktoken-encoder", type=str, default="cl100k_base", help="Tiktoken encoder (default: cl100k_base)")
    parser.add_argument("--faker-langage", type=str, default="fr_FR", help="Langage used for generating prompt responses (default: fr_FR)")
    parser.add_argument("--faker-seed", type=int, default=None, help="Seed for Faker generation (optional)")

    # TEI-specific arguments
    parser.add_argument("--payload-limit", type=int, default=2000000, help="Payload size limit in bytes (default: 2000000)")
    parser.add_argument("--max-client-batch-size", type=int, default=32, help="Maximum number of inputs per request (default: 32)")
    parser.add_argument("--auto-truncate", action="store_true", help="Automatically truncate inputs longer than max size")
    parser.add_argument("--max-batch-tokens", type=int, default=16384, help="Maximum total tokens in a batch (default: 16384)")

    return parser.parse_args()


def create_app(args):
    """Create and configure FastAPI application"""
    if args.api_key:
        settings.api_key = args.api_key
    if args.tiktoken_encoder:
        settings.tiktoken_encoder = args.tiktoken_encoder
    if args.faker_langage:
        settings.faker_langage = args.faker_langage
    if args.faker_seed:
        settings.faker_seed = args.faker_seed

    app = FastAPI(
        title="OpenMockLLM API",
        description="Mock LLM API Server supporting vllm and mistral",
        version="1.0.0",
    )

    # Store configuration in app state
    app.state.backend = args.backend
    app.state.max_context = args.max_context
    app.state.owned_by = args.owned_by
    app.state.model_name = args.model_name
    app.state.embedding_dimension = args.embedding_dimension

    # Include routers based on backend
    if args.backend == "vllm":
        from openmockllm.vllm.endpoints import chat, embeddings, health, models
        from openmockllm.vllm.exceptions import VLLMException, general_exception_handler, vllm_exception_handler

        # Add exception handlers
        app.add_exception_handler(VLLMException, vllm_exception_handler)
        app.add_exception_handler(Exception, general_exception_handler)

        # Add routers (prefixes are defined in the router instances)
        app.include_router(chat.router)
        app.include_router(models.router)
        app.include_router(embeddings.router)
        app.include_router(health.router)
        logger.info("Loaded vllm backend with all endpoints")

    elif args.backend == "mistral":
        from openmockllm.mistral.endpoints import chat, embeddings, models
        from openmockllm.mistral.exceptions import MistralException, general_exception_handler, mistral_exception_handler

        # Add exception handlers
        app.add_exception_handler(MistralException, mistral_exception_handler)
        app.add_exception_handler(Exception, general_exception_handler)

        # Add routers (prefixes are defined in the router instances)
        app.include_router(chat.router)
        app.include_router(models.router)
        app.include_router(embeddings.router)
        logger.info("Loaded mistral backend with exception handling")

    elif args.backend == "tei":
        from openmockllm.tei.endpoints import embeddings, health, info, rerank
        from openmockllm.tei.exceptions import TEIException, general_exception_handler, tei_exception_handler

        # Store TEI-specific config in app state
        app.state.payload_limit = args.payload_limit
        app.state.max_client_batch_size = args.max_client_batch_size
        app.state.auto_truncate = args.auto_truncate
        app.state.max_batch_tokens = args.max_batch_tokens

        # Add exception handlers
        app.add_exception_handler(TEIException, tei_exception_handler)
        app.add_exception_handler(Exception, general_exception_handler)

        # Add routers
        app.include_router(embeddings.router)
        app.include_router(health.router)
        app.include_router(info.router)
        app.include_router(rerank.router)
        logger.info("Loaded TEI backend with all endpoints")

    return app


def main():
    """Main entry point"""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("OpenMockLLM API Server")
    logger.info("=" * 60)
    logger.info(f"Backend:          {args.backend}")
    logger.info(f"Port:             {args.port}")
    logger.info(f"Max Context:      {args.max_context}")
    logger.info(f"Owned By:         {args.owned_by}")
    logger.info(f"Model Name:       {args.model_name}")
    logger.info(f"API Key:          {'Enabled' if args.api_key else 'Disabled'}")
    logger.info(f"Tiktoken encoder: {args.tiktoken_encoder}")
    logger.info(f"Faker langage:    {args.faker_langage}")
    logger.info(f"Faker seed:       {args.faker_seed if args.faker_seed else 'Disabled'}")

    # TEI-specific parameters
    if args.backend == "tei":
        logger.info(f"Payload Limit:    {args.payload_limit}")
        logger.info(f"Max Client Batch: {args.max_client_batch_size}")
        logger.info(f"Auto Truncate:    {args.auto_truncate}")
        logger.info(f"Max Batch Tokens: {args.max_batch_tokens}")

    logger.info("=" * 60)

    app = create_app(args)

    logger.info(f"Starting server on http://0.0.0.0:{args.port}")
    logger.info(f"API documentation: http://0.0.0.0:{args.port}/docs")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
