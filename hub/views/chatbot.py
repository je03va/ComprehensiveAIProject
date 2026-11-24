import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

import ui
from registry import BY_KEY, ROOT

m = BY_KEY["chatbot"]
ui.header(m)

sys.path.insert(0, str(ROOT / "src"))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from article_qa.conversation import ConversationSession, strip_markdown
    from article_qa.gemini_client import GeminiChatClient
    from article_qa.ingest import load_article_text, normalize_articles
except ModuleNotFoundError as e:
    st.error(f"Missing package for the chatbot: {e.name}. Install the hub requirements "
             "(`pip install -r hub/requirements.txt`).")
    st.stop()

# ---------------- Settings ----------------
with st.expander("Settings", expanded=not os.getenv("GEMINI_API_KEY")):
    c1, c2 = st.columns(2)
    key = c1.text_input("Gemini API key", value=os.getenv("GEMINI_API_KEY", ""), type="password",
                        help="Read from AIProjects/.env (GEMINI_API_KEY) if it is set there.")
    model = c2.text_input("Model", os.getenv("GEMINI_MODEL", "gemini-3.6-flash"))
    c1, c2 = st.columns(2)
    speak = c1.toggle("Speak answers (gTTS)", value=False)
    memory = c2.toggle("Follow-up memory", value=False,
                       help="Off: every question is answered on its own, like the original CLI. "
                            "On: earlier questions and answers in this chat are sent along so "
                            "follow-ups like 'why?' work. Answers stay grounded in the source.")

# ---------------- Source ----------------
st.subheader("1. Source")
c1, c2 = st.columns([3, 2])
pasted = c1.text_area("Paste article text", height=180, key="qa_text")
files = c2.file_uploader("...or upload PDF / text files", type=["pdf", "txt", "md"], accept_multiple_files=True)


def build_source():
    parts = [normalize_articles(pasted)] if pasted.strip() else []
    if files:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for f in files:
                p = Path(tmp) / f.name
                p.write_bytes(f.getvalue())
                paths.append(str(p))
            parts.append(load_article_text(paths))
    return normalize_articles(parts)


source = build_source()
if source:
    st.caption(f"Source loaded: {len(source.split()):,} words")
    with st.expander("Preview source"):
        st.write(source[:3000] + (" ..." if len(source) > 3000 else ""))

# ---------------- Chat ----------------
st.subheader("2. Ask")
if st.session_state.get("qa_source") != source:      # new source -> new conversation
    st.session_state.qa_source = source
    st.session_state.qa_history = []

c1, c2 = st.columns([5, 1])
if c2.button("Clear chat"):
    st.session_state.qa_history = []

for turn in st.session_state.qa_history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn.get("audio"):
            st.audio(turn["audio"], format="audio/mp3")

question = st.chat_input("Ask a question about the source", disabled=not source)
if question:
    with st.chat_message("user"):
        st.write(question)
    try:
        session = ConversationSession(source)
        session.ask(question)
        payload = session.build_gemini_payload()
        contents = payload["contents"]
        if memory:
            earlier = [{"role": "model" if t["role"] == "assistant" else "user", "content": t["content"]}
                       for t in st.session_state.qa_history]
            contents = earlier + contents
        with st.spinner("Gemini is reading the source..."):
            answer = GeminiChatClient(api_key=key or None, model=model or None).generate(
                payload["system_instruction"], contents)
        session.record_answer(answer)
        audio = None
        if speak:
            from article_qa.tts import TTSEngine
            with st.spinner("Generating speech..."):
                audio = TTSEngine().synthesize(answer)
        with st.chat_message("assistant"):
            st.write(strip_markdown(answer))
            if audio:
                st.audio(audio, format="audio/mp3", autoplay=True)
        st.session_state.qa_history += [{"role": "user", "content": question},
                                        {"role": "assistant", "content": strip_markdown(answer), "audio": audio}]
    except Exception as e:      # missing key, network, oversized prompt...
        st.error(str(e))
elif not source:
    st.info("Add some source text or a file first.")
