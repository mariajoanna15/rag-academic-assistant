import streamlit as st
import os
from dotenv import load_dotenv
from collections import defaultdict

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

load_dotenv()

DB_PATH = "vectorstore"
DATA_PATH = "data"

os.makedirs(DATA_PATH, exist_ok=True)

# ---------------------------
# Load Vector DB
# ---------------------------
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

# ---------------------------
# Create / Update Vector DB
# ---------------------------
def update_vectorstore():
    docs = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            loaded_docs = loader.load()

            for d in loaded_docs:
                d.metadata["source"] = file
                d.metadata["page"] = d.metadata.get("page", 0)

            docs.extend(loaded_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)

    return db

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Academic RAG Assistant", layout="wide")

# ---------------------------
# HEADER
# ---------------------------
st.markdown("""
<h1 style='text-align: center;'> Academic Research Assistant</h1>
<p style='text-align: center; color: gray;'>
Ask questions from your PDFs (Local AI powered by LLaMA3)
</p>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR (UPLOAD)
# ---------------------------
st.sidebar.markdown("## 📂 Upload PDFs")
st.sidebar.markdown("Drag & drop your research papers")

uploaded_files = st.sidebar.file_uploader(
    " ",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(DATA_PATH, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

    st.sidebar.success(f"✅ {len(uploaded_files)} file(s) uploaded")

    with st.spinner("🔄 Indexing documents... This may take a few seconds"):
        db = update_vectorstore()

# ---------------------------
# SHOW LOADED FILES (CLEAN)
# ---------------------------
if os.path.exists(DATA_PATH):
    files = os.listdir(DATA_PATH)
    clean_files = [f for f in files if not f.startswith(".")]

    if clean_files:
        st.sidebar.markdown("### 📄 Loaded Files")
        st.sidebar.caption(f"{len(clean_files)} documents loaded")

        for f in clean_files:
            st.sidebar.markdown(
                f"""
                <div style="
                    background-color:#1e1e1e;
                    padding:8px;
                    border-radius:8px;
                    margin-bottom:5px;
                    font-size:13px;">
                    {f}
                </div>
                """,
                unsafe_allow_html=True
            )

# ---------------------------
# LOAD DB (NOT FORCED)
# ---------------------------
db = None

if os.path.exists(DB_PATH):
    db = load_db()

# ---------------------------
# LLM
# ---------------------------
llm = Ollama(model="llama3")

if db:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

# ---------------------------
# CHAT HISTORY
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# WELCOME SCREEN
# ---------------------------
if not st.session_state.messages:
    st.markdown("""
    <div style='text-align: center; margin-top: 80px;'>
        <h2> Welcome!</h2>
        <p>You can upload PDFs or ask questions from already indexed documents.</p>
        <p style='color: gray;'>Your AI answers only from your research papers.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# DISPLAY CHAT
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# USER INPUT
# ---------------------------
query = st.chat_input("Ask something about your documents...")

if query:
    if db is None:
        st.error("⚠️ No documents found. Please upload PDFs first.")
        st.stop()

    # User message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.spinner("Thinking..."):
        result = qa(query)

    answer = result["result"]

    # Detect "no answer"
    if "does not contain" in answer.lower() or "don't know" in answer.lower():
        show_sources = False
    else:
        show_sources = True

    # Assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

        if show_sources:
            source_dict = defaultdict(set)

            for doc in result["source_documents"]:
                src = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                source_dict[src].add(page)

            st.markdown("### 📖 Sources")

            for src, pages in source_dict.items():
                pages_list = sorted(list(pages))
                pages_str = ", ".join(str(p) for p in pages_list)

                st.markdown(
                    f"""
                    <div style="
                        background-color:#1e1e1e;
                        padding:10px;
                        border-radius:10px;
                        margin-bottom:5px;">
                        📄 <b>{src}</b> — Pages: {pages_str}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })