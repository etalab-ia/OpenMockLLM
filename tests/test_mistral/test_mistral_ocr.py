import pytest


@pytest.mark.asyncio
async def test_ocr_basic(mistral_client):
    """Test basic OCR request with prompt"""
    response = mistral_client.ocr.process(
        model="openmockllm", document={"type": "document_url", "document_url": "https://arxiv.org/pdf/2201.04234"}, include_image_base64=True
    )

    assert response is not None
    assert response.model == "openmockllm"
    assert response.pages is not None
    assert len(response.pages) > 0

    # Verify page structure
    page = response.pages[0]
    assert page.index is not None
    assert page.markdown is not None
    assert len(page.markdown) > 0
    assert page.images is not None
    assert len(page.images) > 0

    # Verify image structure
    image = page.images[0]
    assert image.id is not None
    assert image.image_base64 is not None
    assert image.top_left_x is not None
    assert image.top_left_y is not None
    assert image.bottom_right_x is not None
    assert image.bottom_right_y is not None

    # Verify dimensions
    assert page.dimensions is not None
    assert page.dimensions.dpi is not None
    assert page.dimensions.height is not None
    assert page.dimensions.width is not None

    # Verify usage info
    assert response.usage_info is not None
    assert response.usage_info.pages_processed > 0
    assert response.usage_info.doc_size_bytes > 0
