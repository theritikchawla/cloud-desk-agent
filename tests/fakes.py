"""Fake helpers so the tests never need the internet, an API key or any quota."""
import zlib

import numpy as np

from gemini_client import GeminiError, MESSAGES
from retrieval import tokenize


class FakeEmbedder:
    """Turns text into a 256-number vector by hashing its words. Counts how often it is called."""

    def __init__(self, fail_kind=None):
        self.calls = 0
        self.fail_kind = fail_kind

    def embed_texts(self, texts):
        self.calls += 1
        if self.fail_kind:
            raise GeminiError(self.fail_kind, MESSAGES[self.fail_kind])
        vectors = []
        for text in texts:
            v = np.zeros(256)
            for word in tokenize(text):
                v[zlib.crc32(word.encode()) % 256] += 1
            vectors.append(v.tolist())
        return vectors


class FakeLLM:
    """Pretends to be Gemini. Records every prompt it receives."""

    def __init__(self, answer="OK", error_kind=None):
        self.answer = answer
        self.error_kind = error_kind
        self.prompts = []
        self.systems = []

    def generate(self, system_instruction, prompt):
        self.prompts.append(prompt)
        self.systems.append(system_instruction)
        if self.error_kind:
            raise GeminiError(self.error_kind, MESSAGES[self.error_kind])
        return self.answer

    @property
    def calls(self):
        return len(self.prompts)
