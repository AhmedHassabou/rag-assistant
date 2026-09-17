import logging

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.query import QueryRequest, QueryResponse, ImageQueryResponse
from app.services.retrieval import retrieval_service
from app.services.generation import generate_answer
from app.services import vision

logger = logging.getLogger("rag_assistant")
router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        chunks = retrieval_service.retrieve(request.question)
        answer, sources = generate_answer(request.question, chunks)
        return QueryResponse(answer=answer, sources=sources)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error handling /query")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/query-image", response_model=ImageQueryResponse)
async def query_image(question: str, image: UploadFile = File(...)):
    """Extended Track: accepts a page image alongside the question.
    Runs YOLO detection on the image and fuses the result into the RAG prompt."""
    try:
        image_bytes = await image.read()
        detections = vision.run_detection(image_bytes)
        detection_text = ", ".join(detections) if detections else "no notable elements detected"

        chunks = retrieval_service.retrieve(question)
        answer, sources = generate_answer(question, chunks, extra_context=detection_text)

        return ImageQueryResponse(answer=answer, sources=sources, detections=detections)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error handling /query-image")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
