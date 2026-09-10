import os

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import tempfile
import time
from pathlib import Path

import streamlit as st

from docrouter.cli import build_chunks_for_pdf
from docrouter.generate import generate_answer
from docrouter.index import build_index, retrieve_hybrid
from docrouter.lexical import build_lexical_index

st.set_page_config(page_title="docrouter", page_icon="📄")
st.title("docrouter - multimodal document Q&A")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.index = None
    st.session_state.model = None
    st.session_state.lexical_index = None
    st.session_state.indexed_files = []

uploaded_files = st.file_uploader(
    "Upload one or more PDFs", type="pdf", accept_multiple_files=True
)

if uploaded_files and st.button("Index documents"):
    # Drop the previous index up front: leaving it in place while a re-index
    # fails leaves the UI answering from documents the user replaced.
    st.session_state.chunks = []
    st.session_state.index = None
    st.session_state.model = None
    st.session_state.lexical_index = None
    st.session_state.indexed_files = []

    all_chunks = []
    start_time = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for uploaded_file in uploaded_files:
                tmp_path = Path(tmp_dir) / uploaded_file.name
                tmp_path.write_bytes(uploaded_file.getvalue())
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    all_chunks.extend(build_chunks_for_pdf(str(tmp_path)))
    except Exception as e:
        st.error(f"Something went wrong while processing your document: {e}")
        st.stop()

    if all_chunks:
        index, model = build_index(all_chunks)
        lexical_index = build_lexical_index(all_chunks)
        elapsed = time.perf_counter() - start_time
        st.session_state.chunks = all_chunks
        st.session_state.index = index
        st.session_state.model = model
        st.session_state.lexical_index = lexical_index
        st.session_state.indexed_files = [f.name for f in uploaded_files]
        st.success(
            f"Indexed {len(all_chunks)} chunks from "
            f"{len(uploaded_files)} file(s) in {elapsed:.2f}s."
        )
    else:
        st.warning("No extractable content found in the uploaded file(s).")

if st.session_state.index is not None:
    st.caption(f"Indexed: {', '.join(st.session_state.indexed_files)}")
    question = st.text_input("Ask a question about your documents")

    if question:
        model = st.session_state.model
        if model is None:
            st.error("The document model is not available.")
            st.stop()

        with st.spinner("Retrieving and generating answer..."):
            retrieved = retrieve_hybrid(
                question,
                st.session_state.chunks,
                st.session_state.index,
                model,
                st.session_state.lexical_index,
                k=3,
            )
            answer = generate_answer(question, retrieved)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Show retrieved context"):
            for score, chunk in retrieved:
                st.markdown(f"**[{score:.3f}]**")
                st.text(chunk)
