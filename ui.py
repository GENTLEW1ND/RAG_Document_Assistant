import streamlit as st
import tempfile
import hashlib
from Main import setup_pipeline, run_pipeline

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("📄 RAG Document Assistant")

st.info("Upload a PDF and ask multiple questions about it.")

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
    st.session_state.vec_chunks = None
    st.session_state.sum_chunks = None
    st.session_state.file_path = None


# -----------------------------
# HELPERS
# -----------------------------
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name


def get_file_hash(uploaded_file):
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


@st.cache_resource
def cached_setup(file_hash, file_path):
    return setup_pipeline(file_path)


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:

    new_hash = get_file_hash(uploaded_file)

    # ✅ Only process if NEW document
    if new_hash != st.session_state.file_hash:

        st.info("New document detected. Processing...")

        file_path = save_uploaded_file(uploaded_file)

        with st.spinner("Processing document (one-time)..."):
            vec_chunks, sum_chunks = cached_setup(new_hash, file_path)

        # Store in session
        st.session_state.file_hash = new_hash
        st.session_state.vec_chunks = vec_chunks
        st.session_state.sum_chunks = sum_chunks
        st.session_state.file_path = file_path

    else:
        st.success("Using existing document ✅")


# -----------------------------
# QUERY SECTION
# -----------------------------
if st.session_state.vec_chunks:

    st.divider()
    st.subheader("🔍 Ask Questions")

    query = st.text_input("Enter your query:")

    if st.button("Search"):

        if not query.strip():
            st.warning("Please enter a query")
        else:
            with st.spinner("Running pipeline..."):

                answer, docs = run_pipeline(
                    query,
                    st.session_state.vec_chunks,
                    st.session_state.sum_chunks
                )

            # -----------------------------
            # OUTPUT
            # -----------------------------
            st.subheader("🧠 Answer")
            st.write(answer)

            # Debug view
            with st.expander("📄 Retrieved Chunks"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(doc.page_content[:400])
                    st.divider()

else:
    st.warning("Upload a document to begin.")


# -----------------------------
# RESET BUTTON (Optional but useful)
# -----------------------------
if st.session_state.vec_chunks:
    if st.button("🔄 Reset Document"):
        st.session_state.file_hash = None
        st.session_state.vec_chunks = None
        st.session_state.sum_chunks = None
        st.session_state.file_path = None
        st.rerun()