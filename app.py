import streamlit as st

# Tab 2: your existing RAG chat function
from chat import answer_query

# Tab 1: document summarizer helpers
from summarizer import extract_text, summarize_document


def main():
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="📄",
        layout="centered",
    )

    st.title("AI Document Assistant")

    tab1, tab2 = st.tabs(["📄 Document Summarizer", "💬 RAG Chat (Knowledge Base)"])

    # ----------------------------
    # TAB 1: Document Summarizer
    # ----------------------------
    with tab1:
        st.subheader("Upload a document and get a clean summary")

        # Initialize state for this tab
        if "last_summary" not in st.session_state:
            st.session_state["last_summary"] = ""
        if "last_text" not in st.session_state:
            st.session_state["last_text"] = ""

        uploaded = st.file_uploader(
            "Upload PDF, DOCX, or TXT",
            type=["pdf", "docx", "txt"],
        )

        text = ""
        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            text = extract_text(uploaded.name, file_bytes)
            st.session_state["last_text"] = text

        col1, col2 = st.columns(2)

        do_summary = False
        with col1:
            do_summary = st.button("Summarize", type="primary", disabled=(uploaded is None))

        with col2:
            st.download_button(
                "Download extracted text",
                data=(st.session_state["last_text"] or "").encode("utf-8", errors="ignore"),
                file_name="extracted_text.txt",
                mime="text/plain",
                disabled=(uploaded is None),
            )

        if do_summary:
            with st.spinner("Summarizing..."):
                st.session_state["last_summary"] = summarize_document(st.session_state["last_text"])

        if st.session_state["last_summary"]:
            st.subheader("Summary")
            st.write(st.session_state["last_summary"])

        if st.session_state["last_text"]:
            with st.expander("Show extracted text"):
                st.write(st.session_state["last_text"])

    # ----------------------------
    # TAB 2: RAG Chat
    # ----------------------------
    with tab2:
        st.subheader("Ask questions using your FAISS knowledge base")

        # Initialize state for chat tab
        if "history" not in st.session_state:
            st.session_state["history"] = []

        user_input = st.text_input("You:", key="input")

        if st.button("Send") and user_input:
            try:
                result = answer_query(user_input)

                # Your answer_query may return dict or string depending on your edits
                if isinstance(result, dict):
                    answer = result.get("answer", "")
                    sources = result.get("sources", [])
                else:
                    answer = str(result)
                    sources = []

            except Exception as e:
                answer = f"Error generating answer: {e}"
                sources = []

            st.session_state["history"].append((user_input, answer, sources))
            st.rerun()

        for user_text, bot_reply, sources in st.session_state["history"]:
            st.markdown(f"**You:** {user_text}")
            st.markdown(f"**Bot:** {bot_reply}")

            if sources:
                with st.expander("Sources used"):
                    for i, s in enumerate(sources, start=1):
                        st.markdown(f"**Source {i}:** {s}")

        st.markdown("---")


if __name__ == "__main__":
    main()
