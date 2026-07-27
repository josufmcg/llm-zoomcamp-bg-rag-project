"""Streamlit metrics dashboard for the BG2 RAG system.

Displays conversation statistics, cost tracking, relevance distribution,
and user feedback summaries. Reads directly from PostgreSQL.

Run with:
    streamlit run src/bg_rag/frontend/dashboard.py --server.port 8502
"""

import os
import sys

import pandas as pd
import streamlit as st

# Add the src directory to the path so we can import bg_rag modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bg_rag.db import get_db_connection


def get_conversations(limit: int = 100) -> pd.DataFrame:
    """Fetch recent conversations from the database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, answer, model, search_method,
                       prompt_tokens, completion_tokens, total_tokens,
                       response_time, cost, timestamp
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([dict(r) for r in rows])


def get_stats() -> dict:
    """Get aggregate conversation statistics."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    COALESCE(AVG(response_time), 0) as avg_response_time,
                    COALESCE(SUM(cost), 0) as total_cost,
                    COALESCE(AVG(total_tokens), 0) as avg_tokens
                FROM conversations
            """)
            row = cur.fetchone()
    finally:
        conn.close()

    return dict(row)


def get_relevance_stats() -> dict:
    """Get judge relevance distribution."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT relevance, COUNT(*) as count
                FROM feedback
                WHERE source = 'judge' AND relevance IS NOT NULL
                GROUP BY relevance
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    return {row["relevance"]: row["count"] for row in rows}


def get_user_feedback_stats() -> dict:
    """Get user feedback counts."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COALESCE(SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END), 0) as thumbs_up,
                    COALESCE(SUM(CASE WHEN score < 0 THEN 1 ELSE 0 END), 0) as thumbs_down
                FROM feedback
                WHERE source = 'user'
            """)
            row = cur.fetchone()
    finally:
        conn.close()

    return dict(row)


def main() -> None:
    st.set_page_config(
        page_title="BG2 RAG Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 BG2 RAG Dashboard")

    # Summary metrics
    stats = get_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Conversations", stats["total"])
    col2.metric("Avg Response Time", f"{stats['avg_response_time']:.2f}s")
    col3.metric("Total Cost", f"${stats['total_cost']:.4f}")
    col4.metric("Avg Tokens", f"{stats['avg_tokens']:.0f}")

    # Charts
    df = get_conversations(limit=100)

    if not df.empty:
        st.subheader("Cost Over Time")
        st.line_chart(df.set_index("timestamp")["cost"])

        st.subheader("Response Time Over Time")
        st.line_chart(df.set_index("timestamp")["response_time"])

        st.subheader("Search Method Distribution")
        method_counts = df["search_method"].value_counts()
        st.bar_chart(method_counts)
    else:
        st.info("No conversations yet. Ask some questions first!")

    # Relevance stats
    st.subheader("Judge Relevance Distribution")
    relevance = get_relevance_stats()
    if relevance:
        st.bar_chart(pd.Series(relevance))
    else:
        st.info("No judge evaluations yet.")

    # User feedback
    st.subheader("User Feedback")
    feedback = get_user_feedback_stats()
    col1, col2 = st.columns(2)
    col1.metric("👍 Thumbs Up", int(feedback["thumbs_up"]))
    col2.metric("👎 Thumbs Down", int(feedback["thumbs_down"]))

    # Recent conversations
    st.subheader("Recent Conversations")
    if not df.empty:
        for _, row in df.head(10).iterrows():
            with st.expander(f"Q: {row['question'][:80]}..."):
                st.write(f"**Answer:** {row['answer'][:300]}...")
                st.write(
                    f"⏱️ {row['response_time']:.2f}s | "
                    f"🔤 {row['total_tokens']} tokens | "
                    f"💰 ${row['cost']:.6f} | "
                    f"🔍 {row['search_method']}"
                )
    else:
        st.info("No conversations yet.")


if __name__ == "__main__":
    main()
