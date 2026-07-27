"""Streamlit conversation UI for the BG2 RAG system.

Provides a chat interface for asking Baldur's Gate II character
creation questions, with metrics display and feedback buttons.

Run with:
    streamlit run src/bg_rag/frontend/app.py --server.port 8501
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


def check_api_health() -> bool:
    """Check if the FastAPI backend is reachable."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.ConnectionError:
        return False


def ask_question(question: str, search_method: str) -> dict | None:
    """Send a question to the FastAPI /ask endpoint.

    Args:
        question: The user's question.
        search_method: "vector", "keyword", or "hybrid".

    Returns:
        Response dict with answer, metrics, and conversation_id.
        None if the request fails.
    """
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "search_method": search_method},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None


def submit_feedback(conversation_id: int, score: int) -> bool:
    """Submit user feedback to the FastAPI /feedback endpoint.

    Args:
        conversation_id: The conversation to rate.
        score: +1 (thumbs up) or -1 (thumbs down).

    Returns:
        True if successful.
    """
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={"conversation_id": conversation_id, "score": score},
            timeout=10,
        )
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def main() -> None:
    st.set_page_config(
        page_title="BG2 Character Guide",
        page_icon="⚔️",
        layout="wide",
    )

    st.title("⚔️ Baldur's Gate II Character Guide")
    st.markdown(
        "Ask questions about character creation, classes, kits, and strategies."
    )

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        search_method = st.selectbox(
            "Search Method",
            options=["hybrid", "vector", "keyword"],
            index=0,
            help="How to search the knowledge base for relevant information.",
        )

        # API health indicator
        if check_api_health():
            st.success("✅ API connected")
        else:
            st.error("❌ API not reachable")
            st.caption(f"Expected at: {API_URL}")

        st.divider()
        st.caption("📊 [Open Dashboard](./dashboard)")

    # Initialize session state
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None

    # Question input
    user_input = st.text_input(
        "Your question:",
        placeholder="e.g., What is the best Fighter kit for dealing damage?",
    )

    if st.button("Ask", type="primary", disabled=not user_input):
        with st.spinner("Thinking..."):
            result = ask_question(user_input, search_method)

        if result:
            st.session_state.conversation_id = result["conversation_id"]
            st.session_state.last_answer = result

            # Display answer
            st.subheader("Answer")
            st.write(result["answer"])

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Response Time", f"{result['response_time']:.2f}s")
            col2.metric("Prompt Tokens", result["prompt_tokens"])
            col3.metric("Completion Tokens", result["completion_tokens"])
            col4.metric("Cost", f"${result['cost']:.6f}")

            # Display relevance (from LLM judge)
            if result.get("relevance"):
                relevance_color = {
                    "RELEVANT": "🟢",
                    "PARTLY_RELEVANT": "🟡",
                    "NON_RELEVANT": "🔴",
                }.get(result["relevance"], "⚪")
                st.markdown(
                    f"**Judge:** {relevance_color} {result['relevance']}"
                )
                if result.get("relevance_explanation"):
                    st.caption(result["relevance_explanation"])

    # Feedback buttons (only show if there's a conversation)
    if st.session_state.conversation_id is not None:
        st.divider()
        st.markdown("**Was this answer helpful?**")
        col1, col2, _ = st.columns([1, 1, 8])

        with col1:
            if st.button("👍", key="thumbs_up"):
                if submit_feedback(st.session_state.conversation_id, 1):
                    st.success("Thanks for the feedback!")
                else:
                    st.error("Failed to submit feedback")

        with col2:
            if st.button("👎", key="thumbs_down"):
                if submit_feedback(st.session_state.conversation_id, -1):
                    st.info("Thanks for the feedback!")
                else:
                    st.error("Failed to submit feedback")


if __name__ == "__main__":
    main()
