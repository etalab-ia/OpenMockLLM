def test_health_check_basic(tei_client):
    """Test basic health check endpoint"""
    response = tei_client.get("/health")

    assert response.status_code == 200


def test_health_check_no_auth(tei_client):
    """Test that health endpoint doesn't require authentication"""
    # Make request without any authentication headers
    response = tei_client.get("/health")

    # Health check should work without authentication
    assert response.status_code == 200
