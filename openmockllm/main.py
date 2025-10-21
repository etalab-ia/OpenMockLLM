import argparse

from fastapi import FastAPI
import uvicorn

from openmockllm.logging import init_logger

logger = init_logger("openmockllm")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="OpenMockLLM - Mock LLM API Server")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "mistral"],
        required=True,
        help="Backend to use (vllm or mistral)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=128000,
        help="Maximum context length (default: 128000)",
    )
    parser.add_argument(
        "--owned-by",
        type=str,
        default="OpenMockLLM",
        help="Owner of the API (default: OpenMockLLM)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openmockllm",
        help="Model name to return (default: openmockllm)",
    )

    return parser.parse_args()


def create_app(args):
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="OpenMockLLM API",
        description="Mock LLM API Server supporting vLLM and Mistral",
        version="1.0.0",
    )

    # Store configuration in app state
    app.state.backend = args.backend
    app.state.max_context = args.max_context
    app.state.owned_by = args.owned_by
    app.state.model_name = args.model_name

    # Include routers based on backend
    if args.backend == "vllm":
        from openmockllm.vllm.endpoints import chat, embeddings, models

        app.include_router(chat.router, prefix="/v1/chat", tags=["chat"])
        app.include_router(models.router, prefix="/v1/models", tags=["models"])
        app.include_router(embeddings.router, prefix="/v1/embeddings", tags=["embeddings"])
        logger.info("Loaded vLLM backend")

    elif args.backend == "mistral":
        from openmockllm.mistral.endpoints import chat, embeddings, models

        app.include_router(chat.router, prefix="/v1/chat", tags=["chat"])
        app.include_router(models.router, prefix="/v1/models", tags=["models"])
        app.include_router(embeddings.router, prefix="/v1/embeddings", tags=["embeddings"])
        logger.info("Loaded Mistral backend")

    @app.get("/")
    async def root():
        return {
            "message": "OpenMockLLM API Server",
            "backend": args.backend,
            "model": args.model_name,
            "max_context": args.max_context,
            "owned_by": args.owned_by,
        }

    @app.get("/health")
    async def health():
        return {"status": "healthy", "backend": args.backend}

    return app


def main():
    """Main entry point"""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("OpenMockLLM API Server")
    logger.info("=" * 60)
    logger.info(f"Backend:      {args.backend}")
    logger.info(f"Port:         {args.port}")
    logger.info(f"Max Context:  {args.max_context}")
    logger.info(f"Owned By:     {args.owned_by}")
    logger.info(f"Model Name:   {args.model_name}")
    logger.info("=" * 60)

    app = create_app(args)

    logger.info(f"Starting server on http://0.0.0.0:{args.port}")
    logger.info(f"API documentation: http://0.0.0.0:{args.port}/docs")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
