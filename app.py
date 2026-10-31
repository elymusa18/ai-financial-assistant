"""AI Financial Data Assistant (Streamlit UI).

Pipeline: CSV -> cleaning (pandas) -> optional auto-categorisation (embeddings + kNN)
-> summary & charts -> chat that answers with RAG (retrieved transactions + summary).
"""
import os

import pandas as pd
import streamlit as st
from openai import OpenAI

from analysis import build_context, compute_summary
from categorizer import categorize_dataframe, evaluate_categorizer
from cleaning import UNKNOWN, load_and_clean
from embeddings import HashingEmbedder, OpenAIEmbedder
from llm import ask_llm
from rag import TransactionIndex

st.set_page_config(page_title="AI Financial Data Assistant", layout="wide")
st.title("AI Financial Data Assistant")

LOCAL, OPENAI = "Offline (hashing, free)", "OpenAI embeddings (semantic)"

with st.sidebar:
    api_key = st.text_input("OpenAI API key", type="password",
                            value=os.getenv("OPENAI_API_KEY", ""))
    model = st.text_input("Chat model", value="gpt-4o-mini")
    embed_mode = st.radio("Embeddings", [LOCAL, OPENAI])
    threshold = st.slider("Categorisation confidence threshold", 0.05, 0.9,
                          0.25 if embed_mode == LOCAL else 0.40, 0.05,
                          help="Predictions below this similarity stay 'Unknown'.")
    auto_categorize = st.checkbox("Auto-categorise 'Unknown' transactions", value=True)
    st.caption("CSV columns: date, category, type (income/expense), amount, description (optional)")

client = OpenAI(api_key=api_key) if api_key else None
if embed_mode == OPENAI and client is None:
    st.sidebar.warning("No API key: using the offline embedder instead.")
embedder = OpenAIEmbedder(client) if (embed_mode == OPENAI and client) else HashingEmbedder()

uploaded = st.file_uploader("Upload a CSV file", type="csv")
if not uploaded:
    st.info("Upload a CSV to start. You can try sample_data.csv.")
    st.stop()

try:
    df, clean_report = load_and_clean(uploaded)
except ValueError as e:
    st.error(str(e))
    st.stop()

with st.expander("Data cleaning report", expanded=False):
    st.json(clean_report)

# ---- Auto-categorisation (embeddings + kNN) ----
if auto_categorize:
    try:
        df, cat_report = categorize_dataframe(df, embedder, threshold=threshold)
    except Exception as e:
        st.error(f"Categorisation failed: {e}")
        st.stop()
    st.subheader("Auto-categorisation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Needed a category", cat_report["to_categorize"])
    c2.metric("Categorised", cat_report["categorized"])
    c3.metric("Left for review", cat_report["left_unknown"])

    reviewed = df[(df["category_predicted"]) | ((df["category"] == UNKNOWN) & (df["description"] != ""))]
    if not reviewed.empty:
        st.caption("Predicted categories with confidence (please review):")
        st.dataframe(reviewed[["date", "description", "amount", "category", "category_confidence"]]
                     .sort_values("category_confidence"), use_container_width=True)

    if st.button("Measure categoriser accuracy (hold-out test)"):
        try:
            result = evaluate_categorizer(df[~df["category_predicted"]], embedder, threshold=threshold)
            st.json({k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()})
        except ValueError as e:
            st.warning(str(e))

# ---- Summary & charts ----
summary = compute_summary(df)
m1, m2, m3 = st.columns(3)
m1.metric("Income", f"{summary['income']:,.2f}")
m2.metric("Expenses", f"{summary['expenses']:,.2f}")
m3.metric("Profit", f"{summary['profit']:,.2f}")

left, right = st.columns(2)
with left:
    st.subheader("Monthly income vs expenses")
    st.bar_chart(summary["monthly"][["income", "expense"]])
with right:
    st.subheader("Expenses by category")
    st.bar_chart(summary["by_category"])

# ---- RAG chat ----
st.subheader("Ask about your data")
data_key = (uploaded.name, uploaded.size, embed_mode if client else LOCAL, auto_categorize, threshold)
if st.session_state.get("data_key") != data_key:
    st.session_state.data_key = data_key
    st.session_state.messages = []
    st.session_state.index = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("e.g. How much did I spend on subscriptions? / Колко похарчих за храна?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            if st.session_state.index is None:
                with st.spinner("Indexing transactions..."):
                    st.session_state.index = TransactionIndex(df, embedder)
            retrieved = st.session_state.index.search(question, k=8)
            with st.expander("Retrieved transactions (RAG)"):
                st.write("\n".join(f"- {r}" for r in retrieved))

            if client is None:
                answer = "Enter your OpenAI API key in the sidebar to get an answer. The retrieved transactions are shown above."
            else:
                with st.spinner("Thinking..."):
                    answer = ask_llm(client, model, build_context(summary), question, retrieved)
        except Exception as e:
            answer = f"Request failed: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
