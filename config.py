"""All settings live here so the other files stay clean.

Secrets (the API key) are read from the .env file and are never printed.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --- Gemini settings (from .env) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2").strip()
DEVELOPER_MODE = os.getenv("DEVELOPER_MODE", "true").strip().lower() == "true"

# --- Local paths ---
KB_DIR = BASE_DIR / "knowledge_base"
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "clouddesk.db")))
CACHE_DIR = BASE_DIR / "embeddings_cache"

# --- Retrieval settings ---
MAX_CHUNK_CHARS = 900   # a chunk is one section of a document, split if longer than this
TOP_K = 3               # how many chunks we hand to Gemini
# Minimum similarity (0 to 1) for a chunk to count as "relevant".
# Embedding scores and keyword (TF-IDF) scores live on different scales.
MIN_SCORE_EMBEDDINGS = float(os.getenv("MIN_SCORE_EMBEDDINGS", "0.60"))
MIN_SCORE_TFIDF = float(os.getenv("MIN_SCORE_TFIDF", "0.12"))

# --- Chat settings ---
MAX_QUESTION_CHARS = 1000
HISTORY_MESSAGES = 6            # how many earlier messages Gemini sees for follow-ups
REQUEST_TIMEOUT_SECONDS = 30


def key_is_set() -> bool:
    """True if a real-looking key is configured. Never returns or prints the key."""
    return bool(GEMINI_API_KEY) and not GEMINI_API_KEY.lower().startswith("your-")
