# Support Triage Agent

A terminal-based AI agent built for the **HackerRank Orchestrate** hackathon. It reads customer support tickets spanning three product ecosystems — **HackerRank**, **Claude**, and **Visa** — and triages each one: classifying the request, retrieving grounded documentation, and deciding whether to reply directly or escalate to a human.

> This is a portfolio snapshot of an in-progress hackathon build, not a polished production service. See [Status](#status--known-limitations) below.

## Problem

Companies receive large volumes of support tickets. Simple, repetitive questions ("How do I reset my password?") shouldn't need a human, while urgent or risky issues (fraud, security, account lockouts) shouldn't be answered with a guess. This agent acts as a first responder: it reads each ticket, checks the official knowledge base for that product, and either answers safely with a grounded response or escalates.

Full task spec: [`problem_statement.md`](./problem_statement.md). Scoring rubric: [`evalutation_criteria.md`](./evalutation_criteria.md).

## How it works

```
support_tickets.csv
        │
        ▼
1. Parse ticket (Subject, Issue, Company)
        │
        ▼
2. Classify & assess (request type, product area, risk/urgency)
        │
        ▼
3. Retrieve knowledge — semantic search over data/ via Chroma
        │
        ▼
4. Decide — reply vs. escalate
        │
        ▼
5. Generate grounded response + justification
        │
        ▼
output.csv
```

- **Retrieval (RAG):** Markdown docs under `data/` are chunked (`langchain-text-splitters`), embedded, and stored in a local Chroma collection (`chroma_db/`). Tickets are answered only from retrieved context, not model parametric knowledge.
- **Per-company routing:** `agents/` holds one agent per product ecosystem (`hackerrank_agent.py`, `claude_agent.py`, `visa_agent.py`) plus a shared base (`support_agent.py`) that handles risk/urgency assessment and escalation logic.
- **LLM:** `models/support_agent_model.py` wraps the configured provider (currently Google Gemini via `langchain-google-genai`); `models/chroma_model.py` wraps the vector store.
- **Prompting:** `src/prompt_builder.py` and `src/ticket_analyzer.py` assemble the per-ticket prompt sent to the model; `src/constants.py` is the single source of truth for prompts, paths, and tunables.

## Repository layout

```
agents/                 # Per-company agent logic + shared SupportAgent base
models/                 # LLM wrapper + Chroma vector store wrapper
pipelines/              # RAG ingestion pipeline + ticket-processing pipeline
src/
  constants.py          # Paths, env vars, prompts, thresholds — no magic values elsewhere
  data_handler.py        # CSV load/write via pandas
  document_handler.py    # Loads & chunks the data/ knowledge base
  metadata_extractor.py   # Pulls company/product-area metadata from docs
  prompt_builder.py      # Builds the per-ticket LLM prompt
  ticket_analyzer.py     # Orchestrates classify -> retrieve -> decide -> respond
  enums.py               # Company, RequestType, Risk, Status, Urgency
  utils.py               # Timers, logging, progress bar, shared helpers
support_tickets/
  support_tickets.csv          # Full input dataset
  sample_support_tickets.csv   # Small dataset with expected outputs, for dev/testing
  output.csv                   # Agent-produced results (evaluator reads this — don't rename)
data/
  claude/                # Claude Help Center export (knowledge base)
  hackerrank/             # HackerRank support articles (knowledge base)
  visa/                   # Visa support articles (knowledge base)
chroma_db/              # Local persisted vector store (generated, gitignored-ready)
tests/                  # Pytest unit tests
main.py                 # Entry point
```

## Setup

Requires Python 3.14.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

| Variable         | Purpose                                    |
|------------------|---------------------------------------------|
| `MODEL_API_KEY`  | LLM provider API key                        |
| `MODEL_API_URL`  | LLM provider base URL (optional)            |
| `MODEL_NAME`     | Model identifier string                     |
| `MODEL_EMBEDDING`| Embedding model identifier                  |
| `HF_TOKEN`       | HuggingFace token (if using HF embeddings)  |
| `APP_NAME`       | Display name for the CLI banner (optional)  |

Never commit `.env` — it's already gitignored.

## Running

```bash
source venv/bin/activate
python main.py --sample --rag     # sample dataset, build/refresh the vector store
python main.py --rag              # full dataset
python main.py --refresh --rag    # force re-ingest data/ into Chroma
```

Recognized flags (`src/constants.py::ARGS_LIST`): `--eda`, `--log`, `--rag`, `--refresh`, `--sample`.

Results are written to `support_tickets/output.csv` with columns: `status`, `product_area`, `response`, `justification`, `request_type`.

## Testing

```bash
pytest
```

## Status & known limitations

This is a work in progress, tracked here for transparency:

- `pipelines/main.py::run_process_tickets_pipeline` currently only processes the first ticket row (`row.Index == 0`) before exiting — full-dataset batch processing is not wired up yet.
- `pipelines/process_tickets.py` and `pipelines/rag.py` are placeholder module headers pending a refactor out of `pipelines/main.py`.
- `src/utils.py` is still being consolidated; some helpers referenced in code (`gen_run_id`, `show_banner`, etc.) live there but are under active revision.
- The LLM provider is being migrated from Gemini toward a HuggingFace-embeddings + configurable-chat-model setup; `models/chroma_model_claude.py` is an in-progress alternate embedding backend.

## Key conventions

- All paths, env vars, and prompts live in `src/constants.py` — don't hardcode them elsewhere.
- `data/` is reference-only knowledge base content — don't modify it.
- `support_tickets/output.csv` and `main.py` are fixed entry points the evaluator relies on — don't rename them.
- Secrets come only from environment variables, never hardcoded.

## License

MIT — see [LICENSE](./LICENSE).
