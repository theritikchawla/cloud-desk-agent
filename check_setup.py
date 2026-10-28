"""Run this once to check your setup:   python check_setup.py

It makes only 2 tiny API requests (one answer, one embedding) and never prints your key.
"""
import config
from gemini_client import GeminiClient, GeminiError


def main():
    print("CloudDesk setup check\n")
    print(f"  Answer model:    {config.GEMINI_MODEL}")
    print(f"  Embedding model: {config.GEMINI_EMBEDDING_MODEL}")

    if not config.key_is_set():
        print("\n[X] No API key found. Open the file named .env and put your key after GEMINI_API_KEY=")
        return

    client = GeminiClient(config.GEMINI_API_KEY, config.GEMINI_MODEL, config.GEMINI_EMBEDDING_MODEL)
    print("\n[1/2] Asking Gemini a tiny question...")
    try:
        print("  [OK] Gemini replied:", client.generate("Answer in one word.", "Reply with the single word: ready"))
    except GeminiError as e:
        print("  [X]", e.user_message)

    print("\n[2/2] Testing the embedding model...")
    try:
        vectors = client.embed_texts(["hello"])
        print(f"  [OK] Embeddings work ({len(vectors[0])} numbers per text).")
    except GeminiError as e:
        print("  [X]", e.user_message)
        print("  The app will still work: it falls back to keyword search automatically.")


if __name__ == "__main__":
    main()
