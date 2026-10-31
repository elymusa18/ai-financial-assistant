"""Prompt construction and the chat-completion call."""

SYSTEM_PROMPT = (
    "You are a careful financial data assistant. Answer ONLY using the data "
    "provided. Do not invent numbers. If the data cannot answer the question, "
    "say so. The retrieved transactions are only the closest matches to the "
    "question, not the full dataset: use the summary for totals and the "
    "retrieved transactions for details. Be concise and answer in the same "
    "language as the question."
)


def build_user_prompt(context: str, question: str, retrieved: list[str] | None = None) -> str:
    parts = [f"SUMMARY:\n{context}"]
    if retrieved:
        lines = "\n".join(f"- {r}" for r in retrieved)
        parts.append("RETRIEVED TRANSACTIONS (date | type | category | amount | description):\n" + lines)
    parts.append(f"QUESTION: {question}")
    return "\n\n".join(parts)


def ask_llm(client, model: str, context: str, question: str,
            retrieved: list[str] | None = None) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(context, question, retrieved)},
        ],
    )
    return response.choices[0].message.content
