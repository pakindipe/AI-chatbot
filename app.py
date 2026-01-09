"""
app.py
------

This Streamlit application provides a simple chat interface for interacting
with the retrieval‑augmented chatbot.  It uses the ``answer_query`` function
from ``chat.py`` to generate responses based on a knowledge base built by
``ingest.py``.

To run the app, first ensure you have run ``ingest.py`` to build the index and
that the environment has the required dependencies installed.  Then execute::

    streamlit run app.py

in the terminal from this directory.  A web browser window will open with the
chat interface.
"""

import streamlit as st
from chat import answer_query


def main() -> None:
    st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")
    st.title("AI Chatbot")
    st.markdown("\nType a question below and press **Send** to get a response.")

    # Initialize conversation history
    if "history" not in st.session_state:
        st.session_state["history"] = []  # list of (user_input, bot_reply, sources)

    # Input box for user query
    user_input = st.text_input("You:", key="input")

    if st.button("Send") and user_input:
        try:
            result = answer_query(user_input)
            answer = result["answer"]
            sources = result["sources"]
        except Exception as e:
            answer = f"Error generating answer: {e}"
            sources = []

        st.session_state["history"].append((user_input, answer, sources))
        st.rerun()

    # Display conversation history
    for user_text, bot_reply, sources in st.session_state["history"]:
        st.markdown(f"**You:** {user_text}")
        st.markdown(f"**Bot:** {bot_reply}\n")

        with st.expander("Sources used"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**Source {i}:** {src}")


if __name__ == "__main__":
    main()
