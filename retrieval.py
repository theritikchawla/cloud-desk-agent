"""Finds the documentation chunks that best match a question.

Two ways to search:
  1. Embeddings (preferred): Gemini turns text into vectors (lists of numbers).
     Texts with similar meaning get similar vectors. We compare the question
     vector with every chunk vector using cosine similarity.
  2. TF-IDF keyword search (fallback): works fully offline with no API.
     It is NOT an embedding model. It scores chunks by shared, rarer words.
     We use it automatically if embeddings are unavailable (no key, quota, error).

Document embeddings are saved on disk, so they are only created once (and again
only if the documents or the embedding model change).
"""
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from gemini_client import GeminiError
from knowledge_base import Chunk

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were", "be",
    "it", "its", "this", "that", "with", "as", "at", "by", "from", "do", "does", "did", "can", "could",
    "i", "me", "my", "we", "you", "your", "our", "how", "what", "which", "who", "when", "where", "why",
    "will", "would", "should", "there", "if", "so", "but", "not", "no", "yes", "have", "has", "had",
    "get", "about", "any", "than", "then", "they", "them", "their", "up", "out", "am", "us",
    "much", "many", "offer", "help", "please", "need", "want", "like", "use", "just", "also", "tell",
}

# Tiny synonym list so "cost" finds "price". (Real embeddings handle this automatically.)
SYNONYMS = {
    "cost": "price", "costs": "price", "pricing": "price", "prices": "price", "fee": "price", "fees": "price",
    "cancellation": "cancel", "cancelling": "cancel", "canceling": "cancel", "cancelled": "cancel",
    "canceled": "cancel", "login": "log", "logins": "log", "signin": "log", "refunds": "refund",
    "refunded": "refund", "passwords": "password",
}


MIN_COVERAGE = 0.5


def _stem(word: str) -> str:
    """Very light stemming so 'prices' matches 'price' and 'billing' matches 'billed'."""
    for suffix in ("ing", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    words = [SYNONYMS.get(w, w) for w in re.findall(r"[a-z0-9]+", text.lower())]
    return [_stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


def cosine_similarity(query_vector, matrix) -> np.ndarray:
    """Cosine similarity: 1.0 = same direction (same meaning), 0 = unrelated."""
    q = np.asarray(query_vector, dtype=float)
    m = np.asarray(matrix, dtype=float)
    q_norm = np.linalg.norm(q)
    m_norms = np.linalg.norm(m, axis=1)
    if q_norm == 0:
        return np.zeros(len(m))
    m_norms[m_norms == 0] = 1.0
    return (m @ q) / (m_norms * q_norm)


class TfidfIndex:
    """A tiny TF-IDF search engine built with NumPy only."""

    def __init__(self, texts: list[str]):
        docs = [tokenize(t) for t in texts]
        self.vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
        n = len(docs)
        df = np.zeros(len(self.vocab))
        for d in docs:
            for w in set(d):
                df[self.vocab[w]] += 1
        # Rare words get a bigger weight than words that appear everywhere.
        self.idf = np.log((1 + n) / (1 + df)) + 1
        self.matrix = np.vstack([self._vector(d) for d in docs]) if docs else np.zeros((0, 0))

    def _vector(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for word, count in Counter(tokens).items():
            if word in self.vocab:
                vec[self.vocab[word]] = (1 + math.log(count)) * self.idf[self.vocab[word]]
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self._vector(tokenize(query))

    def coverage(self, query: str) -> float:
        """Share of the question's words that appear anywhere in the documentation."""
        tokens = tokenize(query)
        return sum(t in self.vocab for t in tokens) / len(tokens) if tokens else 0.0


@dataclass
class Hit:
    chunk: Chunk
    score: float
    method: str  # "embeddings" or "tfidf"
    coverage: float = 1.0  # only used by tfidf (see MIN_COVERAGE)


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder=None, cache_dir=None, embedding_model: str = "",
                 min_score_embeddings: float = config.MIN_SCORE_EMBEDDINGS,
                 min_score_tfidf: float = config.MIN_SCORE_TFIDF, top_k: int = config.TOP_K):
        self.chunks = chunks
        self.embedder = embedder              # needs an .embed_texts(list) method, or None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.embedding_model = embedding_model
        self.min_embeddings = min_score_embeddings
        self.min_tfidf = min_score_tfidf
        self.top_k = top_k
        # The heading is added twice so a matching section title counts for more.
        self.tfidf = TfidfIndex([f"{c.heading} {c.heading} {c.text}" for c in chunks])
        self.matrix = None                    # document vectors (None = embeddings not available)
        self.method = "tfidf"
        self.notice = ""                      # a friendly note if we had to fall back
        self.loaded_from_cache = False

    # ---------- building ----------
    def build(self) -> None:
        """Prepare embeddings (from cache if possible). Falls back to TF-IDF on any API problem."""
        if self.embedder is None:
            self.notice = "Embeddings are off, so keyword search (TF-IDF) is being used."
            return
        try:
            self.matrix = self._load_or_create_embeddings()
            self.method = "embeddings"
        except GeminiError as exc:
            self.matrix = None
            self.method = "tfidf"
            self.notice = f"{exc.user_message} Using keyword search (TF-IDF) instead."

    def _fingerprint(self) -> str:
        data = self.embedding_model + "\n" + "\n".join(c.text for c in self.chunks)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _load_or_create_embeddings(self) -> np.ndarray:
        fingerprint = self._fingerprint()
        if self.cache_dir:
            meta_file, vec_file = self.cache_dir / "meta.json", self.cache_dir / "vectors.npy"
            if meta_file.exists() and vec_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    vectors = np.load(vec_file)
                    if meta.get("fingerprint") == fingerprint and len(vectors) == len(self.chunks):
                        self.loaded_from_cache = True
                        return vectors
                except (ValueError, OSError):
                    pass  # unreadable cache: just rebuild it
        vectors = np.array(self.embedder.embed_texts([c.text for c in self.chunks]), dtype=float)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_dir / "vectors.npy", vectors)
            (self.cache_dir / "meta.json").write_text(
                json.dumps({"fingerprint": fingerprint, "model": self.embedding_model,
                            "count": len(self.chunks)}), encoding="utf-8")
        return vectors

    # ---------- searching ----------
    def rank(self, query: str) -> list[Hit]:
        """The best top_k chunks, with scores, whether or not they are relevant enough."""
        scores, method = None, "tfidf"
        if self.matrix is not None and self.embedder is not None:
            try:
                query_vector = self.embedder.embed_texts([query])[0]
                scores, method = cosine_similarity(query_vector, self.matrix), "embeddings"
            except GeminiError as exc:
                self.notice = f"{exc.user_message} Using keyword search (TF-IDF) for this question."
        if scores is None:
            scores = self.tfidf.scores(query)
        coverage = self.tfidf.coverage(query) if method == "tfidf" else 1.0
        order = np.argsort(-scores)[: self.top_k]
        return [Hit(self.chunks[i], float(scores[i]), method, coverage) for i in order]

    def search(self, query: str) -> list[Hit]:
        """Only the chunks that pass the relevance threshold. Empty list = nothing useful found."""
        hits = self.rank(query)
        return [h for h in hits if self.is_relevant(h)]

    def is_relevant(self, hit: Hit) -> bool:
        if hit.method == "embeddings":
            return hit.score >= self.min_embeddings
        # Keyword search is easily fooled by one shared common word, so we also require that
        # most of the question's words exist in the documentation at all.
        return hit.score >= self.min_tfidf and hit.coverage >= MIN_COVERAGE
