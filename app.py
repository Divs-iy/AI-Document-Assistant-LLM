%%writefile app.py
import streamlit as st
import traceback
import tempfile
import os
import json
from datetime import date

from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


GROQ_API_KEY=os.getenv("GROQ_API_KEY")
APP_PASSWORD="AI"
DAILY_LIMIT=20


st.set_page_config(page_title="Deep Study AI", page_icon="📚", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .main-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(90deg, #6C63FF, #48CAE4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #888; font-size: 1rem; margin-top: 0; margin-bottom: 2rem; }
    .status-box {
        padding: 0.75rem 1rem; border-radius: 10px;
        background: #1a1f2e; border-left: 4px solid #6C63FF;
        margin-bottom: 1rem; color: #ccc; font-size: 0.9rem;
    }
    .answer-box {
        background: #1a1f2e; border-radius: 12px;
        padding: 1.5rem; margin-top: 1rem;
        border: 1px solid #2e3450; color: #e0e0e0;
        line-height: 1.7;
    }
    .stButton > button {
        background: linear-gradient(90deg, #6C63FF, #48CAE4);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 2rem; font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .stTextInput > div > div > input {
        background: #1a1f2e; border: 1px solid #2e3450;
        color: white; border-radius: 8px;
    }
    section[data-testid="stSidebar"] {
        background-color: #13161f;
        border-right: 1px solid #2e3450;
    }
</style>
""", unsafe_allow_html=True)
# ── password gate───────────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("## Deep Study AI")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Wrong password")
    st.stop()

# ── Store in json file ───────────────────────────────────────────────────────────────────
import uuid

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

USAGE_FILE = "usage.json"

def load_usage():
    today = str(date.today())
    user_id = st.session_state.user_id

    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE) as f:
            data = json.load(f)

        if today in data and user_id in data[today]:
            return data[today][user_id]

    return 0


def increment_usage():
    today = str(date.today())
    user_id = st.session_state.user_id

    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE) as f:
            data = json.load(f)
    else:
        data = {}

    if today not in data:
        data[today] = {}

    if user_id not in data[today]:
        data[today][user_id] = 0

    data[today][user_id] += 1

    # ✅ THIS MUST BE INSIDE FUNCTION
    with open(USAGE_FILE, "w") as f:
        json.dump(data, f)

    return data[today][user_id]


usage_today = load_usage()
remaining = DAILY_LIMIT - usage_today

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📚 Deep Study AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a PDF and ask questions about it using AI</p>', unsafe_allow_html=True)
st.divider()
# pwd = st.text_input("Password", type="password")
# if st.button("Login"):
#       if pwd == APP_PASSWORD:
#           st.session_state.authenticated = True
#           st.rerun()
#       else:
#           st.error("❌ Wrong password")
# st.stop()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()

    # api_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...")
    # st.caption("Get a free key at [groq.com](https://groq.com)")

    st.divider()
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    chunk_size = st.slider("Chunk Size", 200, 1000, 500, step=50)
    top_k = st.slider("Top K Results", 1, 10, 5)

    st.divider()
    st.markdown("### 📖 How to use")
    st.markdown("1. Upload a PDF\n2. Wait for processing\n3. Ask any question!")
    st.divider()

    #usage meter
    pct = int((usage_today / DAILY_LIMIT) * 100)
    color = "#6C63FF" if remaining >5 else "#F0a500" if remaining >0 else "#e05c5c"
    st.markdown(f'<p class= "limimt_bar">Daily usage: {usage_today}/{DAILY_LIMIT}questions</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#2e3450;border-radius:8px;height:8px;width:100%">
      <div style="background:{color};width:{pct}%;height:8px;border-radius:8px"></div>
    </div>""", unsafe_allow_html=True)

    if remaining <=0:
      st.error("Daily limits reached. Resets tomorrow!")
    elif remaining <=5:
      st.warning(f"⚠️ Only {remaining} questions left today")
    if st.button(" Logout"):
      st.session_state.authenticated = False
      st.rerun()

# ── Function ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_db(file_bytes, file_name, chunk_size):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    texts = [doc.page_content for doc in chunks]

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(texts, embedding_model)

    os.unlink(tmp_path)

    return db, len(documents), len(chunks)
# ── MAIN ──
try:
    if not uploaded_file:
        st.info("Upload a PDF to start")
        st.stop()

    db, num_pages, num_chunks = build_db(
        uploaded_file.read(),
        uploaded_file.name,
        chunk_size
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("📄 File", uploaded_file.name[:22] + "...")
    col2.metric("📃 Pages", num_pages)
    col3.metric("🧩 Chunks", num_chunks)
    st.divider()

    st.markdown("### 💬 Ask a Question")
    query = st.text_input(
        "",
        placeholder="e.g. What is this document about?",
        label_visibility="collapsed"
    )

    if st.button("🔍 Ask"):
        if not query:
            st.warning("Please type a question first.")

        elif remaining <= 0:
            st.error("❌ Daily limit reached.")

        else:
            client = Groq(api_key=GROQ_API_KEY)

            with st.spinner("🤔 Thinking..."):
                results = db.similarity_search(query, k=top_k)
                context = "\n\n".join([r.page_content for r in results])

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Answer based only on context."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                    ]
                )

            answer = response.choices[0].message.content
            response_text = response.choices[0].message.content

# Split answer + evidence
            if "Evidence:" in response_text:
                answer_part, evidence_part = response_text.split("Evidence:", 1)
            else:
                answer_part = response_text
                evidence_part = ""

            answer = answer_part.replace("Answer:", "").strip()

            evidence_lines = [
                line.strip("- ").strip()
                for line in evidence_part.strip().split("\n")
                if line.strip()
                ]

            increment_usage()

            st.markdown("#### 📝 Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
            st.markdown("#### 📌 Supporting Evidence")
            st.markdown("""
<style>
.evidence-box {
    background: #1a1f2e;
    border-left: 4px solid #48CAE4;
    padding: 0.75rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    color: #ccc;
}
</style>
""", unsafe_allow_html=True)

            for line in evidence_lines:
              st.markdown(f'<div class="evidence-box">{line}</div>', unsafe_allow_html=True)

            with st.expander("📌 View source chunks used"):
                for i, r in enumerate(results):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.caption(r.page_content)

except Exception:
    st.error("Something went wrong")
    st.code(traceback.format_exc())
