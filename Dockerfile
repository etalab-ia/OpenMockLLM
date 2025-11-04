FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /
COPY ./openmockllm/ /openmockllm
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv pip install "."

ENV PATH="/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "openmockllm.main", "--backend", "vllm", "--port", "8000"]