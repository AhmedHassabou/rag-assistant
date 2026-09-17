"""
Basic tests for the /query endpoint.

Run with: pytest -v
Note: requires the vector store to be built (notebooks/rag_pipeline.ipynb)
and Ollama running locally, OR mock retrieval_service/generate_answer for
fully isolated unit tests.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_invalid_input_returns_422():
    # question is too short (< min_length=3) -> validation error
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 422


def test_query_happy_path(monkeypatch):
    # Mock the retrieval + generation so this test doesn't need a live
    # vector store or a running Ollama instance.
    from app.api.routes import query as query_route

    def fake_retrieve(question, top_k=None):
        return [{"chunk_id": "1", "text": "Sample note text.", "document": "notes.pdf", "distance": 0.1}]

    def fake_generate(question, chunks, extra_context=""):
        return "This is a grounded answer. [Source: notes.pdf]", ["notes.pdf"]

    monkeypatch.setattr(query_route.retrieval_service, "retrieve", fake_retrieve)
    monkeypatch.setattr(query_route, "generate_answer", fake_generate)

    response = client.post("/query", json={"question": "What is chunking?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "notes.pdf" in data["sources"]
