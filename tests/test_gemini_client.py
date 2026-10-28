from gemini_client import GeminiClient, GeminiError, classify_error


class FakeAPIError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeModels:
    def __init__(self, error=None, text="Hello", dims=3):
        self.error, self.text, self.calls, self.dims = error, text, 0, dims

    def generate_content(self, model, contents, config):
        self.calls += 1
        if self.error:
            raise self.error
        return Obj(text=self.text)

    def embed_content(self, model, contents):
        self.calls += 1
        if self.error:
            raise self.error
        return Obj(embeddings=[Obj(values=[1.0] * self.dims) for _ in contents])


class FakeSDKClient:
    def __init__(self, **kw):
        self.models = FakeModels(**kw)


def client_with(**kw):
    return GeminiClient("secret-key-123", "model", "embed-model", client=FakeSDKClient(**kw))


def test_error_classification():
    cases = [
        (FakeAPIError(429, "RESOURCE_EXHAUSTED: slow down"), "rate_limit"),
        (FakeAPIError(429, "Quota exceeded ... GenerateRequestsPerDayPerProjectPerModel"), "daily_quota"),
        (FakeAPIError(400, "API key not valid. Please pass a valid API key."), "invalid_key"),
        (FakeAPIError(403, "permission denied"), "invalid_key"),
        (FakeAPIError(404, "model not found"), "model_not_found"),
        (FakeAPIError(503, "overloaded"), "server"),
        (FakeAPIError(504, "deadline"), "timeout"),
        (TimeoutError("timed out"), "timeout"),
        (ConnectionError("no route"), "network"),
        (ValueError("weird"), "other"),
    ]
    for exc, expected in cases:
        assert classify_error(exc).kind == expected, (exc, expected)


def test_generate_returns_text():
    assert client_with(text="  Hi there  ").generate("rules", "question") == "Hi there"


def test_missing_key_gives_friendly_error():
    try:
        GeminiClient("", "model", "embed").generate("rules", "question")
        assert False, "should have raised"
    except GeminiError as e:
        assert e.kind == "no_key" and "GEMINI_API_KEY" in e.user_message


def test_placeholder_key_counts_as_missing():
    assert not GeminiClient("your-key-here", "m", "e").is_configured


def test_failed_request_is_not_retried():
    client = client_with(error=FakeAPIError(429, "RESOURCE_EXHAUSTED"))
    try:
        client.generate("rules", "question")
        assert False, "should have raised"
    except GeminiError as e:
        assert e.kind == "rate_limit"
    assert client._client.models.calls == 1  # exactly one attempt, no wasted quota


def test_api_key_never_appears_in_error_messages():
    client = client_with(error=FakeAPIError(400, "API key not valid: secret-key-123"))
    try:
        client.generate("rules", "question")
    except GeminiError as e:
        assert "secret-key-123" not in e.user_message and "secret-key-123" not in str(e)


def test_empty_answer_is_an_error():
    try:
        client_with(text="").generate("rules", "question")
        assert False, "should have raised"
    except GeminiError as e:
        assert e.kind == "empty"


def test_embed_texts_returns_one_vector_per_text():
    vectors = client_with().embed_texts(["a", "b", "c"])
    assert len(vectors) == 3 and len(vectors[0]) == 3


def test_embed_error_is_friendly():
    try:
        client_with(error=FakeAPIError(429, "PerDay quota")).embed_texts(["a"])
        assert False, "should have raised"
    except GeminiError as e:
        assert e.kind == "daily_quota"
