"""The only file that talks to Google's Gemini API.

Rules this file follows (to protect your free quota and your secrets):
  * Every request is tried ONCE. No automatic retries that waste quota.
  * Errors are turned into short, friendly messages (GeminiError).
  * The API key is never printed, logged, or put in an error message.
  * The google-genai library is imported only when first needed, so the rest of
    the project (and the tests) work even without it.
"""
import time

import config


class GeminiError(Exception):
    """A problem with the Gemini API, with a message that is safe to show users."""

    def __init__(self, kind: str, user_message: str):
        super().__init__(user_message)
        self.kind = kind
        self.user_message = user_message


MESSAGES = {
    "no_key": "No Gemini API key is set. Add GEMINI_API_KEY to your .env file and restart the app.",
    "invalid_key": "Google rejected the API key. Check GEMINI_API_KEY in your .env file.",
    "rate_limit": "The free Gemini rate limit was reached. Please wait about a minute and try again.",
    "daily_quota": ("The free Gemini daily quota is used up. It resets at midnight Pacific time. "
                    "You can still read your previous conversations and tickets."),
    "model_not_found": "The Gemini model name was not found. Check GEMINI_MODEL in your .env file.",
    "timeout": "Gemini took too long to answer. Please try again in a moment.",
    "network": "Could not reach Gemini. Please check your internet connection.",
    "server": "Gemini is temporarily unavailable. Please try again later.",
    "empty": "Gemini returned an empty answer (it may have been blocked). Please rephrase your question.",
    "other": "Something went wrong while contacting Gemini. Please try again later.",
}


def classify_error(exc: Exception) -> GeminiError:
    """Turn any exception from the SDK or the network into a GeminiError."""
    if isinstance(exc, GeminiError):
        return exc

    code = getattr(exc, "code", None)
    if not isinstance(code, int):
        code = getattr(exc, "status_code", None)
    text = str(exc)
    lowered = text.lower()
    type_name = type(exc).__name__.lower()

    if code == 429 or "resource_exhausted" in lowered:
        kind = "daily_quota" if ("perday" in lowered or "per day" in lowered) else "rate_limit"
    elif code in (401, 403) or "api key not valid" in lowered or "api_key_invalid" in lowered:
        kind = "invalid_key"
    elif code == 400 and "api key" in lowered:
        kind = "invalid_key"
    elif code == 404:
        kind = "model_not_found"
    elif code in (408, 504) or "timeout" in type_name or "timed out" in lowered:
        kind = "timeout"
    elif isinstance(code, int) and code >= 500:
        kind = "server"
    elif isinstance(exc, (ConnectionError, OSError)) or "connect" in type_name:
        kind = "network"
    else:
        kind = "other"
    return GeminiError(kind, MESSAGES[kind])


class GeminiClient:
    """Small wrapper with two jobs: generate text, and turn text into embeddings."""

    def __init__(self, api_key: str, model: str, embedding_model: str, client=None):
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self._client = client  # tests pass in a fake client here
        self._one_by_one = False   # becomes True if the model merges a list of texts into ONE embedding
        self.pause_seconds = 0.5   # small pause between single requests, to stay under rate limits

    @property
    def is_configured(self) -> bool:
        return self._client is not None or (bool(self.api_key) and not self.api_key.lower().startswith("your-"))

    def _get_client(self):
        if self._client is None:
            if not self.is_configured:
                raise GeminiError("no_key", MESSAGES["no_key"])
            from google import genai  # imported late on purpose

            self._client = genai.Client(
                api_key=self.api_key,
                http_options={"timeout": config.REQUEST_TIMEOUT_SECONDS * 1000},  # milliseconds
            )
        return self._client

    def generate(self, system_instruction: str, prompt: str) -> str:
        """Ask Gemini for an answer. One attempt only."""
        try:
            client = self._get_client()
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "system_instruction": system_instruction,
                    "temperature": 0.2,          # low = more factual, less creative
                    "max_output_tokens": 1500,
                },
            )
            text = getattr(response, "text", None)
        except Exception as exc:  # noqa: BLE001 - we convert every failure on purpose
            raise self._safe(exc) from None
        if not text or not text.strip():
            raise GeminiError("empty", MESSAGES["empty"])
        return text.strip()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Turn each text into a list of numbers (a "vector") that captures its meaning."""
        vectors: list[list[float]] = []
        try:
            client = self._get_client()
            for start in range(0, len(texts), 50):  # at most 50 texts per request
                vectors.extend(self._embed_batch(client, texts[start:start + 50]))
        except Exception as exc:  # noqa: BLE001
            raise self._safe(exc) from None
        if len(vectors) != len(texts):
            raise GeminiError("other", MESSAGES["other"])
        return vectors

    def _embed_batch(self, client, batch: list[str]) -> list[list[float]]:
        """Some embedding models return one vector per text; others (e.g. gemini-embedding-2)
        merge a whole list into a single vector. We detect that and send the texts one at a time."""
        if not self._one_by_one:
            response = client.models.embed_content(model=self.embedding_model, contents=batch)
            vectors = [list(e.values) for e in response.embeddings]
            if len(vectors) == len(batch):
                return vectors
            self._one_by_one = True
        vectors = []
        for i, text in enumerate(batch):
            if i:
                time.sleep(self.pause_seconds)
            response = client.models.embed_content(model=self.embedding_model, contents=text)
            vectors.append(list(response.embeddings[0].values))
        return vectors

    def _safe(self, exc: Exception) -> GeminiError:
        """Classify the error; make sure the key can never appear in the message."""
        error = classify_error(exc)
        if self.api_key and self.api_key in error.user_message:
            return GeminiError(error.kind, MESSAGES["other"])
        return error
