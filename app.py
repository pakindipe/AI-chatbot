import streamlit as st

from chat import answer_query
from summarizer import extract_text, summarize_document


def main():
    st.set_page_config(
        page_title="AI Document Chatbot",
        page_icon="🤖",
        layout="centered",
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
      font-family: 'Poppins', sans-serif;
    }

    h1 { font-weight: 700; }

    section.main > div { padding-top: 2rem; }

    .stButton > button {
      border-radius: 12px;
      padding: 0.55rem 1rem;
      font-weight: 600;
    }

    div[data-testid="stFileUploader"] {
      padding: 0.75rem;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.12);
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("AI Document ChatBot")
    st.caption("By Philip Akindipe")

    tab1, tab2 = st.tabs(["📝 Summarizer", "💬 Chat (Knowledge Base)"])


    # =======================
    # TAB 1: DOCUMENT SUMMARY
    # =======================
    with tab1:
        st.subheader("Upload a document and get a clean summary")

        uploaded = st.file_uploader(
            "Upload PDF, DOCX, or TXT",
            type=["pdf", "docx", "txt"],
        )

        # Initialize state
        st.session_state.setdefault("extracted_text", None)
        st.session_state.setdefault("summary_result", None)

        if uploaded:
            text = extract_text(uploaded.name, uploaded.getvalue())
            st.session_state["extracted_text"] = text

            col1, col2 = st.columns(2)

            with col1:
                do_summary = st.button("Summarize", type="primary")

            with col2:
                st.download_button(
                    "Download extracted text",
                    data=text.encode("utf-8", errors="ignore"),
                    file_name="extracted_text.txt",
                    mime="text/plain",
                )

            if do_summary:
                with st.spinner("Summarizing document..."):
                    st.session_state["summary_result"] = summarize_document(text)

        # ---- Display results ----
        if st.session_state["summary_result"]:
            result = st.session_state["summary_result"]

            st.subheader("Document type")
            st.write(result["doc_type"])

            st.subheader("Summary")
            st.write(result["summary"])

        if st.session_state["extracted_text"]:
            with st.expander("Show extracted text"):
                st.write(st.session_state["extracted_text"])

    # =======================
    # TAB 2: RAG CHAT
    # =======================
    with tab2:
        st.subheader("Ask questions using your FAISS knowledge base")

        st.session_state.setdefault("history", [])

        user_input = st.text_input("You:", key="input")

        if st.button("Send") and user_input:
            try:
                result = answer_query(user_input)

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