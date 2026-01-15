import httpx
import pytest


@pytest.fixture
def health_client():
    """Create an httpx client for testing health endpoint"""
    client = httpx.Client(base_url="http://localhost:8000", timeout=30.0)
    yield client
    client.close()


def test_health_endpoint(health_client):
    """Test health check endpoint"""
    response = health_client.get("/health")

    assert response.status_code == 200
