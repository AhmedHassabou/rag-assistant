# 📚 Study Notes RAG Assistant (Extended Track)

A RAG-powered web app for chatting with your study notes: ask a question, get a
grounded answer cited from your own documents — plus an image-upload mode that
runs YOLO on scanned notes pages and fuses the detections into the answer.

## Overview

- **Domain:** Study notes (lecture PDFs, summary sheets, scanned diagram pages)
- **Track:** Extended (text RAG + Computer Vision/YOLO component)
- **Pipeline:** clean & chunk → embed → store in Chroma → retrieve → prompt a local
  Ollama LLM → grounded, cited answer
- **Vision:** YOLO detects diagram/table/figure elements on an uploaded page image;
  detections are fused into the RAG prompt as extra context

## Architecture

```
┌─────────────┐      question (+ optional image)      ┌──────────────────┐
│  Streamlit  │ ─────────────────────────────────────► │  FastAPI backend │
│  frontend   │ ◄───────────────────────────────────── │   (/query, ...)  │
└─────────────┘        answer + sources (+ detections)  └────────┬─────────┘
                                                                  │
                                        ┌─────────────────────────┼─────────────────────────┐
                                        ▼                         ▼                         ▼
                              ┌──────────────────┐     ┌───────────────────┐      ┌──────────────────┐
                              │ Chroma vector     │     │  Ollama local LLM  │      │  YOLO (ultralytics)│
                              │ store (persisted) │     │  (e.g. llama3.2)   │      │  vision model      │
                              └──────────────────┘     └───────────────────┘      └──────────────────┘
                                        ▲
                                        │ built once in
                              ┌──────────────────┐
                              │ notebooks/        │
                              │ rag_pipeline.ipynb│
                              └──────────────────┘
```

## Tech Stack

| Layer       | Tech                                                             |
|-------------|-------------------------------------------------------------------|
| Notebook    | Jupyter, pandas, pypdf, sentence-transformers, chromadb, ultralytics |
| Backend     | FastAPI, Pydantic v2, Chroma, Ollama Python client                |
| Frontend    | Streamlit                                                          |
| Vector DB   | Chroma (persisted to disk)                                        |
| LLM         | Ollama (local), default model `llama3.2`                          |
| Vision      | YOLO (ultralytics) — Extended Track                                |

## Project Structure

```
rag-assistant-project/
├── notebooks/
│   └── rag_pipeline.ipynb      # build & evaluate the RAG pipeline
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI app, CORS, startup loading
│   │   ├── api/routes/query.py # GET /health, POST /query, POST /query-image
│   │   ├── core/config.py      # settings from .env
│   │   ├── schemas/query.py    # request/response models
│   │   ├── services/           # retrieval.py, generation.py, vision.py
│   │   └── utils/logging_config.py
│   ├── data/vector_store/      # persisted vector store (from the notebook)
│   ├── tests/test_query.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── .env.example
│   └── requirements.txt
├── data/
│   ├── raw_docs/                # your study-notes source documents (not committed)
│   └── images/                  # scanned pages/diagrams for the vision component
├── .gitignore
└── README.md
```

## Domain / Data Description

> TODO: describe your actual study-notes corpus once collected — subject(s), number
> of documents, source (your own notes / course PDFs / public lecture slides), and
> how to obtain them if not committed to the repo (e.g. a shared Drive link).

## Setup

### 0. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally, with a model pulled:
  ```bash
  ollama pull llama3.2
  ```
- Git

### 1. Build the vector store (once)

```bash
cd notebooks
jupyter notebook rag_pipeline.ipynb
# Run all cells top-to-bottom (Kernel → Restart & Run All).
# This persists the vector store into backend/data/vector_store/.
```

### 2. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # adjust values if needed
uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
```

### 3. Frontend

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
# App: http://localhost:8501
```

## Environment Variables

### Backend (`backend/.env`)

| Variable            | Default                       | Description                              |
|----------------------|--------------------------------|-------------------------------------------|
| `VECTOR_STORE_DIR`   | `data/vector_store`            | Path to the persisted Chroma store        |
| `COLLECTION_NAME`    | `study_notes`                  | Chroma collection name                    |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2`           | Sentence-Transformers model               |
| `OLLAMA_MODEL`       | `llama3.2`                     | Local LLM served by Ollama                |
| `OLLAMA_HOST`        | `http://localhost:11434`       | Ollama server address                     |
| `TOP_K`              | `4`                             | Number of chunks retrieved per query      |
| `FRONTEND_ORIGIN`    | `http://localhost:8501`        | Allowed CORS origin                       |
| `YOLO_MODEL_PATH`    | `data/vision/best.pt`          | Path to the YOLO checkpoint               |
| `ENABLE_VISION`      | `true`                         | Toggle the Extended Track vision endpoint |

### Frontend (`frontend/.env`)

| Variable       | Default                 | Description               |
|-----------------|--------------------------|----------------------------|
| `API_BASE_URL`  | `http://localhost:8000` | Backend base URL           |

## API Reference

### `GET /health`

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok"}
```

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is chunking in RAG?"}'
```
```json
{
  "answer": "Chunking splits documents into smaller pieces before embedding... [Source: chunking_notes.pdf]",
  "sources": ["chunking_notes.pdf"]
}
```

### `POST /query-image` (Extended Track)

```bash
curl -X POST "http://localhost:8000/query-image?question=What%20is%20shown%20in%20this%20diagram%3F" \
  -F "image=@page_12.png"
```
```json
{
  "answer": "The diagram illustrates... [Source: lecture3.pdf]",
  "sources": ["lecture3.pdf"],
  "detections": ["diagram (0.91)", "table (0.77)"]
}
```


## Notes on Reproducibility

- Clone into a fresh folder and follow only this README — if any step fails, the
  README needs fixing.
- The notebook must run top-to-bottom via **Kernel → Restart & Run All**.
- `data/raw_docs/`, `data/images/`, and the vector store are excluded from git (see
  `.gitignore`) — see "Domain / Data Description" above for how to obtain them.
