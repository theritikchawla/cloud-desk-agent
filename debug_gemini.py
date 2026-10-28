r"""Diagnostic: shows the REAL technical error behind "Something went wrong".
Run:  .\.venv\Scripts\python.exe debug_gemini.py
Your API key is hidden in the output, so it is safe to copy and paste it into the chat.
"""
import importlib.metadata

import config


def scrub(text: str) -> str:
    key = config.GEMINI_API_KEY
    return text.replace(key, "***KEY***") if key else text


def show_error(e: Exception) -> None:
    print("  [X] type:", type(e).__name__)
    print("      code:", getattr(e, "code", None))
    print("      message:", scrub(str(e))[:600])


print("google-genai version:", end=" ")
try:
    print(importlib.metadata.version("google-genai"))
except Exception as e:  # noqa: BLE001
    print("NOT INSTALLED", type(e).__name__)
print("answer model:   ", config.GEMINI_MODEL)
print("embedding model:", config.GEMINI_EMBEDDING_MODEL)
print("key found:      ", config.key_is_set(), "| key length:", len(config.GEMINI_API_KEY))

from google import genai

client = genai.Client(api_key=config.GEMINI_API_KEY)

print("\n[1] Answer test")
try:
    r = client.models.generate_content(model=config.GEMINI_MODEL, contents="Reply with the single word: ready")
    print("  [OK]", (r.text or "").strip())
except Exception as e:  # noqa: BLE001
    show_error(e)

print("\n[2] Embedding test")
try:
    r = client.models.embed_content(model=config.GEMINI_EMBEDDING_MODEL, contents=["hello"])
    print("  [OK] numbers per text:", len(r.embeddings[0].values))
except Exception as e:  # noqa: BLE001
    show_error(e)

print("\n[3] Models your key can use for embeddings")
try:
    found = False
    for m in client.models.list():
        if "embedContent" in (getattr(m, "supported_actions", None) or []):
            print("   ", m.name)
            found = True
    if not found:
        print("    (none listed)")
except Exception as e:  # noqa: BLE001
    show_error(e)
