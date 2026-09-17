from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User's question about the study notes")


class SourceChunk(BaseModel):
    document: str
    chunk_id: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


class ImageQueryResponse(BaseModel):
    answer: str
    sources: List[str]
    detections: Optional[List[str]] = None
