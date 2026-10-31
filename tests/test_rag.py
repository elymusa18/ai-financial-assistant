import io
from types import SimpleNamespace

import pytest

from analysis import build_context, compute_summary
from cleaning import load_and_clean
from embeddings import HashingEmbedder
from llm import SYSTEM_PROMPT, ask_llm, build_user_prompt
from rag import TransactionIndex

CSV = """date,category,type,amount,description
2026-01-01,Subscriptions,expense,10,Netflix subscription
2026-01-03,Groceries,expense,40,Lidl groceries
2026-01-04,Utilities,expense,60,Electricity bill
2026-01-05,Education,expense,25,Udemy Python course
"""


@pytest.fixture
def index():
    df, _ = load_and_clean(io.StringIO(CSV))
    return TransactionIndex(df, HashingEmbedder())


def test_search_returns_most_relevant_transaction_first(index):
    assert "Netflix" in index.search("netflix", k=2)[0]
    assert "Electricity" in index.search("electricity bill", k=1)[0]


def test_search_respects_k_and_empty_query(index):
    assert len(index.search("groceries", k=3)) == 3
    assert index.search("   ") == []


def test_documents_contain_structured_fields(index):
    assert "2026-01-01 | expense | Subscriptions | 10.00 | Netflix subscription" in index.documents


class FakeChatClient:
    def __init__(self):
        self.kwargs = None
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])


def test_prompt_includes_summary_retrieved_transactions_and_question():
    prompt = build_user_prompt("Total income: 1", "How much on Netflix?", ["row-1", "row-2"])
    assert "Total income: 1" in prompt and "- row-1" in prompt and "How much on Netflix?" in prompt


def test_prompt_without_retrieval_has_no_retrieved_section():
    assert "RETRIEVED" not in build_user_prompt("ctx", "q")


def test_ask_llm_sends_system_prompt_and_returns_text(index):
    df, _ = load_and_clean(io.StringIO(CSV))
    client = FakeChatClient()
    reply = ask_llm(client, "test-model", build_context(compute_summary(df)),
                    "Netflix?", retrieved=index.search("netflix", k=1))
    assert reply == "answer"
    messages = client.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "Netflix subscription" in messages[1]["content"]
    assert client.kwargs["model"] == "test-model"
