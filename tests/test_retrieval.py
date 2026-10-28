import config
from fakes import FakeEmbedder
from knowledge_base import chunk_document, load_and_chunk, load_documents, split_long_text
from retrieval import Retriever, TfidfIndex, cosine_similarity


def make_retriever(**kwargs):
    r = Retriever(load_and_chunk(), **kwargs)
    r.build()
    return r


# ---------- loading and chunking ----------
def test_loads_all_five_documents():
    sources = {d["source"] for d in load_documents(config.KB_DIR)}
    assert sources == {"product_features.md", "billing_policy.md", "account_help.md", "troubleshooting.md", "faq.md"}


def test_chunks_have_source_and_heading():
    chunks = load_and_chunk()
    assert len(chunks) >= 30
    for c in chunks:
        assert c.source.endswith(".md") and c.heading and c.text.strip()
    assert len({c.chunk_id for c in chunks}) == len(chunks)  # ids are unique


def test_chunk_document_splits_at_headings():
    md = "# Title\n\nIntro text.\n\n## One\nFirst section.\n\n## Two\nSecond section."
    chunks = chunk_document("x.md", md)
    assert [c.heading for c in chunks] == ["Overview", "One", "Two"]
    assert "Second section." in chunks[2].text


def test_long_sections_are_split():
    body = "\n\n".join(f"Paragraph number {i} " + "word " * 30 for i in range(10))
    pieces = split_long_text(body, 300)
    assert len(pieces) > 1 and all(len(p) <= 400 for p in pieces)


# ---------- similarity ----------
def test_cosine_similarity_basics():
    scores = cosine_similarity([1, 0], [[1, 0], [0, 1], [-1, 0]])
    assert round(scores[0], 3) == 1.0 and round(scores[1], 3) == 0.0 and round(scores[2], 3) == -1.0


def test_tfidf_prefers_matching_words():
    index = TfidfIndex(["billing and refunds", "password reset help", "mobile app issues"])
    assert int(index.scores("how do I reset my password").argmax()) == 1


# ---------- representative questions (keyword search) ----------
QUESTIONS = [
    ("How do I reset my password?", "account_help.md"),
    ("How much does the Pro plan cost?", "billing_policy.md"),
    ("Can I get a refund on my annual plan?", "billing_policy.md"),
    ("Why am I not getting notification emails?", "troubleshooting.md"),
    ("Does CloudDesk have a timeline view?", "product_features.md"),
    ("What is the maximum file size I can upload on the Free plan?", "product_features.md"),
    ("Can I export my data?", "faq.md"),
    ("How many failed logins before my account is locked?", "account_help.md"),
    ("Do you have a desktop app?", "faq.md"),
    ("How do I cancel my subscription?", "billing_policy.md"),
    ("Is there a free trial?", "billing_policy.md"),
    ("Which payment methods do you accept?", "billing_policy.md"),
    ("How do I contact support?", "troubleshooting.md"),
]


def test_keyword_search_finds_the_right_document():
    r = make_retriever()
    misses = []
    for question, expected in QUESTIONS:
        sources = [h.chunk.source for h in r.search(question)]
        if expected not in sources:
            misses.append((question, expected, sources))
    assert not misses, misses


def test_irrelevant_questions_return_nothing():
    r = make_retriever()
    for question in ["What is the capital of France?", "Write me a poem about pizza",
                     "What's the weather in Rome today?", "Who won the football match yesterday?",
                     "How do I bake bread?", "Do you offer student discounts?"]:
        assert r.search(question) == [], question


def test_refund_question_returns_refund_section():
    r = make_retriever()
    headings = [h.chunk.heading for h in r.search("Can I get a refund on my annual plan?")]
    assert "Refund policy" in headings


# ---------- embeddings and the cache ----------
def test_embeddings_are_cached_on_disk(tmp_path):
    chunks = load_and_chunk()
    first_embedder = FakeEmbedder()
    first = Retriever(chunks, embedder=first_embedder, cache_dir=tmp_path, embedding_model="m1")
    first.build()
    assert first.method == "embeddings" and first_embedder.calls == 1 and not first.loaded_from_cache

    second_embedder = FakeEmbedder()
    second = Retriever(chunks, embedder=second_embedder, cache_dir=tmp_path, embedding_model="m1")
    second.build()
    assert second.loaded_from_cache and second_embedder.calls == 0  # no new API call


def test_cache_is_rebuilt_when_model_or_documents_change(tmp_path):
    chunks = load_and_chunk()
    Retriever(chunks, embedder=FakeEmbedder(), cache_dir=tmp_path, embedding_model="m1").build()

    other_model = FakeEmbedder()
    Retriever(chunks, embedder=other_model, cache_dir=tmp_path, embedding_model="m2").build()
    assert other_model.calls == 1

    changed_docs = FakeEmbedder()
    Retriever(chunks[:-1], embedder=changed_docs, cache_dir=tmp_path, embedding_model="m2").build()
    assert changed_docs.calls == 1


def test_embedding_search_works_with_fake_embedder(tmp_path):
    embedder = FakeEmbedder()
    r = Retriever(load_and_chunk(), embedder=embedder, cache_dir=tmp_path, embedding_model="m",
                  min_score_embeddings=0.3)
    r.build()
    hits = r.search("How do I reset my password?")
    assert hits and hits[0].method == "embeddings" and hits[0].chunk.source == "account_help.md"
    assert r.search("What is the capital of France?") == []


def test_falls_back_to_keyword_search_when_embeddings_fail(tmp_path):
    r = Retriever(load_and_chunk(), embedder=FakeEmbedder(fail_kind="daily_quota"), cache_dir=tmp_path)
    r.build()
    assert r.method == "tfidf" and "quota" in r.notice.lower()
    assert r.search("How do I reset my password?")  # still works


def test_query_embedding_failure_uses_keyword_search_for_that_question(tmp_path):
    embedder = FakeEmbedder()
    r = Retriever(load_and_chunk(), embedder=embedder, cache_dir=tmp_path, embedding_model="m")
    r.build()
    embedder.fail_kind = "rate_limit"
    hits = r.search("How do I reset my password?")
    assert hits and hits[0].method == "tfidf"
