"""
Application settings, loaded from environment variables / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Vector store ---
    VECTOR_STORE_DIR: str = "data/vector_store"
    COLLECTION_NAME: str = "study_notes"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # --- LLM (Ollama) ---
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_HOST: str = "http://localhost:11434"

    # --- Retrieval ---
    TOP_K: int = 4

    # --- CORS ---
    FRONTEND_ORIGIN: str = "http://localhost:8501"

    # --- Vision (Extended Track) ---
    YOLO_MODEL_PATH: str = "data/vision/best.pt"
    ENABLE_VISION: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
