# AI Financial Data Assistant

Streamlit app that cleans a CSV of financial transactions, **auto-categorises**
uncategorised ones with embeddings, and answers natural-language questions with
**RAG** (retrieved transactions + computed summary) through an LLM API.

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...                         # or paste it in the sidebar
streamlit run app.py
```
Try `sample_data.csv` (contains dirty rows and 10 transactions with no category).

## Tests
```bash
pip install -r requirements-dev.txt
pytest
```
Tests use the offline embedder and fake API clients, so they run without a key or network.

## Architecture
| File | Responsibility |
|---|---|
| `cleaning.py` | CSV loading, type/date/amount parsing, duplicates, invalid rows, cleaning report |
| `analysis.py` | Income / expenses / profit, monthly table, expenses by category, LLM context text |
| `embeddings.py` | `HashingEmbedder` (offline, deterministic) and `OpenAIEmbedder` (semantic) behind one interface |
| `categorizer.py` | kNN categoriser over embeddings, dataframe integration, hold-out evaluation |
| `rag.py` | Transactions as documents, embedding index, top-k retrieval |
| `llm.py` | Prompt construction and chat-completion call |
| `app.py` | Streamlit UI (upload, categorisation review, charts, chat) |

## Design decisions
- **Aggregates + retrieved rows, not the whole file.** The LLM gets exact totals computed by pandas
  (models are unreliable at arithmetic) plus the top-k most relevant transactions. This keeps prompts
  small (tokens / context window) and reduces hallucinated numbers. The prompt says retrieved rows are
  a sample, not the full dataset.
- **Categorisation by similarity, not by asking the LLM per row.** Already-labelled transactions are the
  training set; unknown ones get the label of their nearest labelled descriptions. It is cheap, explainable
  (each prediction has a confidence) and can abstain.
- **Abstain instead of guess.** Predictions below a confidence threshold stay `Unknown` for human review.
- **Swappable embedder.** Same interface for offline and OpenAI embeddings, so tests and demos are free.

## Measured results (offline embedder, sample data, 30% hold-out)
| Metric | Value |
|---|---|
| Accuracy when the model answers | 100% |
| Coverage (share it dares to answer) | ~69% |
| Overall accuracy (Unknown counts as wrong) | ~69% |

The offline embedder is lexical, so merchants it never saw (e.g. "iCloud storage plan" when no similar
description was in training) stay `Unknown`. The OpenAI embedder is meant to raise coverage; compare both
on your own data with the "Measure categoriser accuracy" button before claiming a number.

## Limitations
- Sample data is small and synthetic; real bank exports are messier.
- Thousands separators such as `1,234.50` are not parsed (the row is reported as invalid).
- Categories are learned only from the user's own labels; there is no fixed taxonomy.
- Retrieval is top-k, so questions needing every matching row rely on the summary instead.

## AI-assisted development log (fill in honestly)
- Tool used (Copilot / Cursor / other):
- What I delegated:
- What I reviewed and changed:
- A bug or bad suggestion the AI produced and how I caught it:

## Roadmap
- [ ] SQLite storage and SQL queries
- [ ] Compare offline vs OpenAI embeddings on a larger labelled set
- [ ] Per-type categories (income vs expense) and a user-editable category list
