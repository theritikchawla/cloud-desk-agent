"""SQLite storage: conversations, messages, tickets and feedback.

SQLite is a database stored in a single file (clouddesk.db). No server needed.

Safety notes for beginners:
  * We always pass values with "?" placeholders, never by gluing text into SQL.
    This prevents "SQL injection".
  * Every read of conversations/messages is filtered by session_id, so one
    session can never read another session's chat.
"""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'message',
    sources_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    conversation_id TEXT,
    question TEXT NOT NULL,
    summary TEXT NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL UNIQUE REFERENCES messages(id),
    session_id TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating IN (0, 1)),
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path):
        self.path = str(path)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # ---------- conversations ----------
    def create_conversation(self, session_id: str, title: str = "New conversation") -> str:
        conversation_id = uuid.uuid4().hex
        with self._conn() as conn:
            conn.execute("INSERT INTO conversations (id, session_id, title, created_at) VALUES (?, ?, ?, ?)",
                         (conversation_id, session_id, title[:60], _now()))
        return conversation_id

    def conversation_belongs_to(self, conversation_id: str, session_id: str) -> bool:
        if not conversation_id:
            return False
        with self._conn() as conn:
            row = conn.execute("SELECT 1 FROM conversations WHERE id = ? AND session_id = ?",
                               (conversation_id, session_id)).fetchone()
        return row is not None

    def list_conversations(self, session_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id, title, created_at FROM conversations "
                                "WHERE session_id = ? ORDER BY created_at DESC, rowid DESC", (session_id,)).fetchall()
        return [dict(r) for r in rows]

    def set_title_if_new(self, conversation_id: str, title: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE conversations SET title = ? WHERE id = ? AND title = 'New conversation'",
                         (title[:60], conversation_id))

    # ---------- messages ----------
    def add_message(self, conversation_id: str, role: str, content: str, kind: str = "message",
                    sources: list[dict] | None = None) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, kind, sources_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (conversation_id, role, content, kind, json.dumps(sources or []), _now()))
            return cursor.lastrowid

    def get_messages(self, conversation_id: str, session_id: str) -> list[dict]:
        """Messages of one conversation, only if it belongs to this session."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT m.id, m.role, m.content, m.kind, m.sources_json, m.created_at "
                "FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                "WHERE m.conversation_id = ? AND c.session_id = ? ORDER BY m.id", (conversation_id, session_id)).fetchall()
        messages = []
        for r in rows:
            item = dict(r)
            item["sources"] = json.loads(item.pop("sources_json"))
            messages.append(item)
        return messages

    # ---------- tickets ----------
    def create_ticket(self, session_id: str, conversation_id: str | None, question: str, summary: str,
                      priority: str, status: str = "open") -> str:
        with self._conn() as conn:
            while True:
                ticket_id = "CD-" + uuid.uuid4().hex[:8].upper()
                exists = conn.execute("SELECT 1 FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
                if not exists:
                    break
            conn.execute("INSERT INTO tickets (ticket_id, session_id, conversation_id, question, summary, "
                         "priority, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (ticket_id, session_id, conversation_id, question, summary, priority, status, _now()))
        return ticket_id

    def list_tickets(self, session_id: str | None = None) -> list[dict]:
        """All tickets (developer view), or only one session's tickets."""
        with self._conn() as conn:
            if session_id is None:
                rows = conn.execute("SELECT * FROM tickets ORDER BY created_at DESC, rowid DESC").fetchall()
            else:
                rows = conn.execute("SELECT * FROM tickets WHERE session_id = ? ORDER BY created_at DESC, rowid DESC",
                                    (session_id,)).fetchall()
        return [dict(r) for r in rows]

    # ---------- feedback ----------
    def set_feedback(self, message_id: int, session_id: str, rating: int) -> bool:
        """rating: 1 = helpful, 0 = not helpful. Only the owner of an assistant message may rate it."""
        if rating not in (0, 1):
            raise ValueError("rating must be 0 or 1")
        with self._conn() as conn:
            owner = conn.execute(
                "SELECT 1 FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                "WHERE m.id = ? AND c.session_id = ? AND m.role = 'assistant'", (message_id, session_id)).fetchone()
            if owner is None:
                return False
            conn.execute(
                "INSERT INTO feedback (message_id, session_id, rating, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(message_id) DO UPDATE SET rating = excluded.rating, created_at = excluded.created_at",
                (message_id, session_id, rating, _now()))
        return True

    def get_feedback(self, message_id: int, session_id: str) -> int | None:
        with self._conn() as conn:
            row = conn.execute("SELECT rating FROM feedback WHERE message_id = ? AND session_id = ?",
                               (message_id, session_id)).fetchone()
        return None if row is None else row["rating"]

    def feedback_stats(self) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS total, COALESCE(SUM(rating), 0) AS helpful FROM feedback").fetchone()
        total, helpful = row["total"], row["helpful"]
        return {"total": total, "helpful": helpful, "not_helpful": total - helpful,
                "helpful_percent": round(100 * helpful / total, 1) if total else 0.0}

    def recent_not_helpful(self, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT f.message_id, f.created_at, m.content FROM feedback f JOIN messages m ON m.id = f.message_id "
                "WHERE f.rating = 0 ORDER BY f.id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
