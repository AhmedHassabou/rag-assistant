"""
Thin wrapper around the backend API. The base URL is read from an
environment variable — never hard-coded — so the frontend can point
to localhost during dev or a deployed URL in production.
"""
import os

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def check_health() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def ask_question(question: str) -> dict:
    r = requests.post(f"{API_BASE_URL}/query", json={"question": question}, timeout=60)
    r.raise_for_status()
    return r.json()


def ask_question_with_image(question: str, image_bytes: bytes, filename: str) -> dict:
    files = {"image": (filename, image_bytes, "image/png")}
    params = {"question": question}
    r = requests.post(f"{API_BASE_URL}/query-image", params=params, files=files, timeout=90)
    r.raise_for_status()
    return r.json()
