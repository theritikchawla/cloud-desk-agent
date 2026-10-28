"""The Streamlit web interface. Run it with:  streamlit run app.py

This file only draws the screen. All the real work happens in support_service.py.
"""
import re
import uuid

import streamlit as st

import config
from database import Database
from gemini_client import GeminiClient
from knowledge_base import load_and_chunk
from retrieval import Retriever
from support_service import SupportService

st.set_page_config(page_title="CloudDesk Support Assistant", page_icon="💬", layout="wide")


@st.cache_resource(show_spinner="Loading the CloudDesk knowledge base...")
def get_service() -> SupportService:
    """Built once and reused. The first start may make one embedding request; later starts use the saved cache."""
    db = Database(config.DB_PATH)
    db.init()
    llm = GeminiClient(config.GEMINI_API_KEY, config.GEMINI_MODEL, config.GEMINI_EMBEDDING_MODEL)
    retriever = Retriever(load_and_chunk(), embedder=llm if config.key_is_set() else None,
                          cache_dir=config.CACHE_DIR, embedding_model=config.GEMINI_EMBEDDING_MODEL)
    retriever.build()
    return SupportService(db, retriever, llm, api_key=config.GEMINI_API_KEY)


service = get_service()


# ---------- session handling ----------
def get_session_id() -> str:
    """Each browser session gets a random ID. It is kept in the page address so a refresh keeps your history.
    (Anyone with the address can see that history - fine for a local demo, not for production.)"""
    if "session_id" not in st.session_state:
        sid = st.query_params.get("sid", "")
        if not re.fullmatch(r"[0-9a-f]{32}", str(sid)):
            sid = uuid.uuid4().hex
            st.query_params["sid"] = sid
        st.session_state.session_id = sid
    return st.session_state.session_id


SESSION_ID = get_session_id()
st.session_state.setdefault("conversation_id", None)
st.session_state.setdefault("page", "Chat")


# ---------- small helpers (button callbacks) ----------
def new_conversation():
    st.session_state.conversation_id = None
    st.session_state.page = "Chat"


def open_conversation(conversation_id: str):
    st.session_state.conversation_id = conversation_id
    st.session_state.page = "Chat"


def give_feedback(message_id: int, rating: int):
    service.db.set_feedback(message_id, SESSION_ID, rating)


def request_human():
    reply = service.escalate(SESSION_ID, st.session_state.conversation_id)
    st.session_state.conversation_id = reply.conversation_id
    st.session_state.page = "Chat"


def render_sources(sources: list):
    if sources:
        with st.expander("Sources used"):
            for s in sources:
                st.write(f"📄 **{s['source']}** — {s['heading']}  (match score {s['score']})")


def render_message(m: dict, is_last: bool):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] != "assistant":
            return
        render_sources(m["sources"])
        if m["kind"] == "answer":
            rating = service.db.get_feedback(m["id"], SESSION_ID)
            if rating is None:
                c1, c2, _ = st.columns([1, 1, 6])
                c1.button("👍 Helpful", key=f"up_{m['id']}", on_click=give_feedback, args=(m["id"], 1))
                c2.button("👎 Not helpful", key=f"down_{m['id']}", on_click=give_feedback, args=(m["id"], 0))
            else:
                st.caption("Thanks for your feedback: " + ("helpful 👍" if rating == 1 else "not helpful 👎"))
        if is_last and m["kind"] in ("no_answer", "error", "blocked"):
            st.button("🎫 Create a support ticket", key=f"ticket_{m['id']}", on_click=request_human)


# ---------- sidebar ----------
with st.sidebar:
    st.title("💬 CloudDesk Support")
    st.radio("Go to", ["Chat", "My conversations"] + (["Developer"] if config.DEVELOPER_MODE else []), key="page")
    st.button("➕ New conversation", on_click=new_conversation)
    st.button("🙋 Talk to a human", on_click=request_human)
    st.divider()
    st.caption(f"Model: `{config.GEMINI_MODEL}`")
    if service.retriever.method == "embeddings":
        st.caption("Search: Gemini embeddings ✅")
    else:
        st.caption("Search: keyword search (TF-IDF fallback)")
    if not config.key_is_set():
        st.warning("No Gemini API key found. You can read old chats, but new answers need a key in your .env file.")
    if service.retriever.notice:
        st.info(service.retriever.notice)

st.info("🧪 **Portfolio prototype.** CloudDesk is a fictional company. Tickets are saved locally and are not sent to a real "
        "support team. Please don't type real personal data: on the Gemini free tier, Google may use submitted content "
        "to improve its products.")

page = st.session_state.page

# ---------- Chat page ----------
if page == "Chat":
    st.header("Ask CloudDesk Support")
    conversation_id = st.session_state.conversation_id
    messages = service.db.get_messages(conversation_id, SESSION_ID) if conversation_id else []
    if not messages:
        st.write("Hi! Ask me about accounts, billing, features or troubleshooting. For example:")
        st.markdown("- *How do I reset my password?*\n- *How much does the Pro plan cost?*\n"
                    "- *Can I get a refund on an annual plan?*\n- *Why am I not receiving notifications?*")
    for i, m in enumerate(messages):
        render_message(m, is_last=(i == len(messages) - 1))

    question = st.chat_input("Type your question...", max_chars=config.MAX_QUESTION_CHARS)
    if question:
        with st.spinner("Searching the documentation..."):
            reply = service.handle_message(SESSION_ID, st.session_state.conversation_id, question)
        st.session_state.conversation_id = reply.conversation_id
        st.rerun()

# ---------- History page ----------
elif page == "My conversations":
    st.header("My conversations")
    st.caption("Only conversations from this browser session are shown.")
    conversations = service.db.list_conversations(SESSION_ID)
    if not conversations:
        st.write("No conversations yet. Start one on the Chat page.")
    for conv in conversations:
        with st.expander(f"{conv['title']}  ·  {conv['created_at'][:16].replace('T', ' ')} UTC"):
            for m in service.db.get_messages(conv["id"], SESSION_ID):
                st.markdown(f"**{'You' if m['role'] == 'user' else 'Assistant'}:** {m['content']}")
                render_sources(m["sources"])
            st.button("Continue this conversation", key=f"open_{conv['id']}", on_click=open_conversation,
                      args=(conv["id"],))
    my_tickets = service.db.list_tickets(SESSION_ID)
    if my_tickets:
        st.subheader("My demo tickets")
        st.dataframe([{k: t[k] for k in ("ticket_id", "priority", "status", "summary", "created_at")} for t in my_tickets])

# ---------- Developer page ----------
elif page == "Developer" and config.DEVELOPER_MODE:
    st.header("Developer view")
    st.warning("Demo only: this page shows data from ALL sessions. A real product would need admin login.")

    st.subheader("Demo support tickets")
    tickets = service.db.list_tickets()
    st.caption("Stored locally in SQLite. Not sent to a real support team.")
    if tickets:
        st.dataframe([{k: v for k, v in t.items() if k not in ("session_id", "conversation_id")} for t in tickets])
    else:
        st.write("No tickets yet.")

    st.subheader("Customer feedback")
    stats = service.db.feedback_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ratings", stats["total"])
    c2.metric("Helpful", stats["helpful"])
    c3.metric("Not helpful", stats["not_helpful"])
    c4.metric("Helpful %", f"{stats['helpful_percent']}%")
    for item in service.db.recent_not_helpful():
        st.write(f"👎 {item['content'][:200]}")

    st.subheader("Search tester")
    st.caption("Type a question to see which documents match and their scores. "
               "(With Gemini embeddings active, this uses one embedding request.)")
    test_question = st.text_input("Test question")
    if test_question:
        for hit in service.retriever.rank(test_question):
            label = "✅ relevant" if service.retriever.is_relevant(hit) else "❌ below threshold"
            st.write(f"{label} · score {hit.score:.3f} · {hit.method} · **{hit.chunk.source}** — {hit.chunk.heading}")
