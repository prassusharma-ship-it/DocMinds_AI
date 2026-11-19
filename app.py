
import streamlit as st
import hashlib
import html

from src.pdf_processing import extract_pdf, create_chunks
from src.vector_database import VectorDatabase
from src.retrieval import HybridRetriever
from src.rag_pipeline import generate_answer


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# -----------------------------------
# CUSTOM DARK THEME
# -----------------------------------

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background:
            radial-gradient(
                ellipse at 55% 20%,
                #30131f 0%,
                #1b191b 55%,
                #171717 100%
            );
        color: #f5f5f5;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Title */
    .app-title {
        font-size: 43px;
        font-weight: 750;
        letter-spacing: -1.5px;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .app-subtitle {
        color: #a9a0a5;
        font-size: 14px;
        margin-bottom: 30px;
    }

    /* Panel titles */
    .panel-heading {
        font-size: 23px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 15px;
    }

    /* Two main Streamlit panels */
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        background: rgba(25, 23, 25, 0.65);
        border: 1px solid #40363c;
        border-radius: 16px;
        padding: 22px;
        min-width: 0;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 1px solid #393539;
        border-radius: 13px;
        padding: 14px;
        background: #1b1b1c;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: none;
    }

    /* Buttons */
    .stButton > button {
        background: #38232e;
        color: #ffffff;
        border: 1px solid #684653;
        border-radius: 11px;
        min-height: 42px;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: #573544;
        border-color: #a66d83;
        color: white;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 12px;
        border: 1px solid #393238;
        background: rgba(45, 35, 41, 0.65);
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) {
        background: #181618;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        border-radius: 13px;
        border: 1px solid #493941;
        background: #211b20;
    }

    [data-testid="stChatInput"] textarea {
        color: white;
    }

    /* Document card */
    .document-card {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 13px 14px;
        border: 1px solid #383237;
        background: #211d20;
        border-radius: 11px;
        margin-bottom: 10px;
        font-size: 14px;
        overflow-wrap: anywhere;
    }

    .doc-icon {
        color: #c69bac;
        font-size: 20px;
    }

    /* Status text */
    .status-text {
        font-size: 13px;
        color: #b9a9b1;
        padding: 8px 0;
    }

    /* Sources */
    [data-testid="stExpander"] {
        border: 1px solid #42363d;
        background: #211c20;
        border-radius: 11px;
    }

    /* Hide default footer */
    footer {
        visibility: hidden;
    }

    @media (max-width: 768px) {
        .app-title {
            font-size: 32px;
        }

        .block-container {
            padding-top: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------
# SESSION STATE
# -----------------------------------

defaults = {
    "retriever": None,
    "messages": [],
    "indexed_files": [],
    "indexed_signature": None,
    "chunk_count": 0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------------
# HEADER
# -----------------------------------

st.markdown(
    '<div class="app-title">♧ DocuMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'Your intelligent document research assistant'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------------
# DOCUMENT PROCESSING FUNCTION
# -----------------------------------

def process_documents(files):
    documents = []

    for file in files:
        documents.extend(extract_pdf(file))

    chunks = create_chunks(
        documents,
        chunk_size=500,
        overlap=100
    )

    if not chunks:
        raise ValueError(
            "No extractable text found in the PDFs."
        )

    db = VectorDatabase()
    db.build_index(chunks)

    retriever = HybridRetriever(db)

    return retriever, len(chunks)


def get_file_signature(files):
    """Detect whether the selected PDF collection changed."""
    digest = hashlib.sha256()

    for file in sorted(files, key=lambda f: f.name):
        content = file.getvalue()
        digest.update(file.name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)

    return digest.hexdigest()


# -----------------------------------
# MAIN TWO-COLUMN DASHBOARD
# -----------------------------------

left_col, right_col = st.columns(
    [1, 1],
    gap="medium"
)


# ===================================
# LEFT PANEL — KNOWLEDGE BASE
# ===================================

with left_col:

    st.markdown(
        '<div class="panel-heading">Knowledge Base</div>',
        unsafe_allow_html=True
    )

    if st.session_state.indexed_files:

        for filename in st.session_state.indexed_files:
            safe_name = html.escape(filename)

            st.markdown(
                f"""
                <div class="document-card">
                    <span class="doc-icon">▤</span>
                    <span>{safe_name}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.caption("No documents indexed yet.")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF documents."
    )

    if uploaded_files:
        st.caption(
            f"{len(uploaded_files)} file(s) selected"
        )

    if st.button(
        "Process Documents",
        use_container_width=True,
        type="primary"
    ):

        if not uploaded_files:
            st.warning("Please upload a PDF first.")

        else:
            with st.spinner(
                "Extracting, embedding and indexing..."
            ):
                try:
                    signature = get_file_signature(
                        uploaded_files
                    )

                    retriever, count = process_documents(
                        uploaded_files
                    )

                    st.session_state.retriever = retriever
                    st.session_state.chunk_count = count
                    st.session_state.indexed_files = [
                        file.name for file in uploaded_files
                    ]
                    st.session_state.indexed_signature = (
                        signature
                    )
                    st.session_state.messages = []

                    st.success(
                        f"Successfully indexed {count} chunks!"
                    )

                except Exception as error:
                    st.error(
                        f"Document processing failed: {error}"
                    )

    if st.session_state.retriever is not None:

        st.markdown(
            '<div class="status-text">'
            '● Knowledge base ready'
            '</div>',
            unsafe_allow_html=True
        )

        st.caption(
            f"{len(st.session_state.indexed_files)} PDFs"
            f" · {st.session_state.chunk_count} chunks"
        )

        if uploaded_files:
            current_signature = get_file_signature(
                uploaded_files
            )

            if (
                current_signature
                != st.session_state.indexed_signature
            ):
                st.warning(
                    "Your selected files changed. "
                    "Click Process Documents to update the index."
                )

    if st.button(
        "Clear Knowledge Base",
        use_container_width=True
    ):
        st.session_state.retriever = None
        st.session_state.indexed_files = []
        st.session_state.indexed_signature = None
        st.session_state.chunk_count = 0
        st.session_state.messages = []
        st.rerun()


# ===================================
# RIGHT PANEL — AI ASSISTANT
# ===================================

with right_col:

    st.markdown(
        '<div class="panel-heading">AI Assistant</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.messages:

        with st.chat_message("assistant"):
            st.write(
                "Welcome to DocuMind AI! "
                "Upload your PDFs and ask me questions."
            )

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])

            if message.get("sources"):

                with st.expander("View Sources"):

                    for i, chunk in enumerate(
                        message["sources"],
                        start=1
                    ):
                        st.markdown(
                            f"**[{i}] "
                            f"{chunk['source']} — "
                            f"Page {chunk['page']}**"
                        )

                        st.caption(chunk["text"])

    question = st.chat_input(
        "Ask a question about your documents..."
    )

    if question:

        if st.session_state.retriever is None:
            st.warning(
                "Please upload and process PDFs first."
            )

        else:

            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching documents..."
                ):

                    try:
                        chunks = (
                            st.session_state.retriever.search(
                                question,
                                top_k=5
                            )
                        )

                        answer = generate_answer(
                            question,
                            chunks
                        )

                        st.write(answer)

                        with st.expander("View Sources"):

                            for i, chunk in enumerate(
                                chunks,
                                start=1
                            ):
                                st.markdown(
                                    f"**[{i}] "
                                    f"{chunk['source']} — "
                                    f"Page {chunk['page']}**"
                                )

                                st.caption(chunk["text"])

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": chunks
                        })

                    except Exception as error:

                        st.error(
                            f"Unable to answer: {error}"
                        )
