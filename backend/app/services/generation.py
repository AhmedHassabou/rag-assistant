"""
Generation service.

Builds a grounded prompt from retrieved chunks + the user's question,
calls the local Ollama LLM, and returns an answer that cites its sources.
"""
import logging
from typing import List, Dict, Tuple

import ollama

from app.core.config import settings

logger = logging.getLogger("rag_assistant")


PROMPT_TEMPLATE = """You are a study-notes assistant. Answer the question using ONLY the
context below. If the answer is not contained in the context, say you don't know —
never invent information.

When you use a piece of context, cite it inline like [Source: <document>].

Context:
{context}

Question: {question}

Answer (with inline citations):"""


def _build_context(chunks: List[Dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c['document']} | chunk {c['chunk_id']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, chunks: List[Dict], extra_context: str = "") -> Tuple[str, List[str]]:
    """Returns (answer_text, list_of_source_documents)."""
    if not chunks:
        return "I couldn't find anything relevant in the study notes to answer that.", []

    context = _build_context(chunks)
    if extra_context:
        context += f"\n\n---\n\n[Image detections]\n{extra_context}"

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    client = ollama.Client(host=settings.OLLAMA_HOST)
    response = client.chat(
        model=settings.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response["message"]["content"]

    sources = sorted({c["document"] for c in chunks})
    return answer, sources
