from database import Database


def make_db(tmp_path):
    db = Database(tmp_path / "test.db")
    db.init()
    return db


def test_save_and_load_conversation(tmp_path):
    db = make_db(tmp_path)
    conv = db.create_conversation("session-a")
    first = db.add_message(conv, "user", "How do I reset my password?")
    second = db.add_message(conv, "assistant", "Click Forgot password.", kind="answer",
                            sources=[{"source": "account_help.md", "heading": "Password reset", "score": 0.5}])
    messages = db.get_messages(conv, "session-a")
    assert [m["id"] for m in messages] == [first, second]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["sources"][0]["source"] == "account_help.md"
    assert messages[0]["created_at"]  # timestamp saved


def test_sessions_cannot_see_each_others_conversations(tmp_path):
    db = make_db(tmp_path)
    conv = db.create_conversation("session-a")
    db.add_message(conv, "user", "private question")
    assert db.get_messages(conv, "session-b") == []
    assert db.list_conversations("session-b") == []
    assert not db.conversation_belongs_to(conv, "session-b")
    assert db.conversation_belongs_to(conv, "session-a")
    assert len(db.list_conversations("session-a")) == 1


def test_title_is_set_once(tmp_path):
    db = make_db(tmp_path)
    conv = db.create_conversation("s")
    db.set_title_if_new(conv, "First question")
    db.set_title_if_new(conv, "Second question")
    assert db.list_conversations("s")[0]["title"] == "First question"


def test_create_and_list_tickets(tmp_path):
    db = make_db(tmp_path)
    t1 = db.create_ticket("s1", "c1", "Question one", "Summary one", "high")
    t2 = db.create_ticket("s2", "c2", "Question two", "Summary two", "low")
    assert t1 != t2 and t1.startswith("CD-")
    tickets = db.list_tickets()
    assert len(tickets) == 2
    ticket = next(t for t in tickets if t["ticket_id"] == t1)
    assert ticket["status"] == "open" and ticket["priority"] == "high"
    assert ticket["question"] == "Question one" and ticket["summary"] == "Summary one" and ticket["created_at"]
    assert [t["ticket_id"] for t in db.list_tickets("s2")] == [t2]


def test_feedback_is_saved_and_can_be_changed(tmp_path):
    db = make_db(tmp_path)
    conv = db.create_conversation("s")
    answer = db.add_message(conv, "assistant", "An answer", kind="answer")
    assert db.get_feedback(answer, "s") is None
    assert db.set_feedback(answer, "s", 1)
    assert db.get_feedback(answer, "s") == 1
    assert db.set_feedback(answer, "s", 0)  # changed their mind
    assert db.get_feedback(answer, "s") == 0
    assert db.feedback_stats()["total"] == 1


def test_feedback_statistics(tmp_path):
    db = make_db(tmp_path)
    conv = db.create_conversation("s")
    ids = [db.add_message(conv, "assistant", f"Answer {i}", kind="answer") for i in range(4)]
    for message_id, rating in zip(ids, [1, 1, 1, 0]):
        db.set_feedback(message_id, "s", rating)
    stats = db.feedback_stats()
    assert stats == {"total": 4, "helpful": 3, "not_helpful": 1, "helpful_percent": 75.0}
    assert db.recent_not_helpful()[0]["content"] == "Answer 3"


def test_feedback_stats_when_empty(tmp_path):
    assert make_db(tmp_path).feedback_stats()["total"] == 0


def test_other_session_cannot_rate_a_message(tmp_path):
    db = make_db(tmp_path)
    conv = db.create_conversation("owner")
    answer = db.add_message(conv, "assistant", "Answer", kind="answer")
    assert db.set_feedback(answer, "intruder", 1) is False
    assert db.feedback_stats()["total"] == 0


def test_user_messages_cannot_be_rated(tmp_path):
    db = make_db(tmp_path)
    conv = db.create_conversation("s")
    question = db.add_message(conv, "user", "A question")
    assert db.set_feedback(question, "s", 1) is False


def test_sql_injection_text_is_stored_harmlessly(tmp_path):
    db = make_db(tmp_path)
    conv = db.create_conversation("s'; DROP TABLE messages; --")
    db.add_message(conv, "user", "x'); DROP TABLE messages; --")
    assert len(db.get_messages(conv, "s'; DROP TABLE messages; --")) == 1
