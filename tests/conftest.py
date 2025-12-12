from mistralai import Mistral
import pytest_asyncio


@pytest_asyncio.fixture
async def mistral_client():
    """Create an official MistralClient SDK instance configured for testing"""

    client = Mistral(api_key=None, server_url="http://localhost:8000")
    yield client
