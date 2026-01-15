from fastapi import APIRouter, Depends, Request
from mistralai.models import OCRImageObject, OCRPageDimensions, OCRPageObject, OCRRequest, OCRResponse, OCRUsageInfo

from openmockllm.mistral.utils.common import check_model_not_found
from openmockllm.security import check_api_key
from openmockllm.utils import generate_unstreamed_chat_content, get_base64_jpeg_image

router = APIRouter(prefix="/v1", tags=["models"])


@router.post(path="/ocr", dependencies=[Depends(dependency=check_api_key)])
async def ocr(request: Request, body: OCRRequest) -> OCRResponse:
    check_model_not_found(called_model=body.model, current_model=request.app.state.model_name)

    content = await generate_unstreamed_chat_content(prompt="Lorem ipsum dolor sit amet, consectetur adipiscing elit.", max_tokens=1000)
    image = get_base64_jpeg_image()

    pages = content.split("\n\n")
    pages = [
        OCRPageObject(
            index=i,
            markdown=page,
            images=[
                OCRImageObject(
                    id="img-0.jpeg",
                    top_left_x=294,
                    top_left_y=220,
                    bottom_right_x=1404,
                    bottom_right_y=649,
                    image_base64=image,
                    image_annotation=None,
                )
            ],
            dimensions=OCRPageDimensions(dpi=200, height=2200, width=1700),
        )
        for i, page in enumerate(pages)
    ]
    response = OCRResponse(
        pages=pages,
        model=request.app.state.model_name,
        usage_info=OCRUsageInfo(pages_processed=len(pages), doc_size_bytes=len(content) * 1024),
        document_annotation=None,
    )
    return response
