import httpx
from mistralai import Mistral
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
