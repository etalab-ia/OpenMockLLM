import httpx
from mistralai import Mistral
from openai import AsyncOpenAI, OpenAI
import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def mistral_client():
    """Create an official MistralClient SDK instance configured for testing"""

    client = Mistral(api_key=None, server_url="http://localhost:8000")

    yield client


@pytest.fixture
def tei_client():
    """Create an httpx client for testing TEI backend"""
    client = httpx.Client(base_url="http://localhost:8000", timeout=30.0)

    yield client

    client.close()


@pytest.fixture
def vllm_client():
    """Create an OpenAI client for testing vLLM backend (OpenAI compatible)"""
    client = OpenAI(api_key="test-key", base_url="http://localhost:8000/v1")

    yield client


@pytest_asyncio.fixture
async def vllm_async_client():
    """Create an async OpenAI client for testing vLLM backend"""
    client = AsyncOpenAI(api_key="test-key", base_url="http://localhost:8000/v1")

    yield client

    await client.close()
