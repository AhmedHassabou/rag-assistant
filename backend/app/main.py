from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes.query import router as query_router
from app.services.retrieval import retrieval_service
from app.services import vision
from app.utils.logging_config import setup_logging

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the vector store and LLM connection ONCE
    logger.info("Starting up RAG assistant backend...")
    retrieval_service.load()
    vision.load_vision_model()
    yield
    logger.info("Shutting down RAG assistant backend...")


app = FastAPI(
    title="Study Notes RAG Assistant",
    description="RAG-powered API for chatting with your study notes (Extended Track: with CV/YOLO support).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
