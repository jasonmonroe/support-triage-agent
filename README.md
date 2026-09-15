# Support Triage Agent

A terminal-based AI agent built for the **HackerRank Orchestrate** hackathon. It reads customer support tickets spanning three product ecosystems — **HackerRank**, **Claude**, and **Visa** — and triages each one: classifying the request, retrieving grounded documentation, verifying and scoring the drafted answer, running company-specific compliance checks, and deciding whether to reply directly or escalate to a human.

> This is a portfolio snapshot of an in-progress hackathon build, not a polished production service. See [Status & known limitations](#status--known-limitations) below.

## Problem

Companies receive large volumes of support tickets. Simple, repetitive questions ("How do I reset my password?") shouldn't need a human, while urgent or risky issues (fraud, security, account lockouts, PCI-DSS-sensitive requests) shouldn't be answered with a guess. This agent acts as a first responder: it reads each ticket, checks the official knowledge base for that product, and either answers safely with a grounded response or escalates.

Full task spec: [`problem_statement.md`](./problem_statement.md). Scoring rubric: [`evalutation_criteria.md`](./evalutation_criteria.md).

## How it works

```
support_tickets.csv
        │
        ▼
1. Load agent for the ticket's company (HackerRank / Claude / Visa / generic)
        │
        ▼
2. Classify & assess — request type, product area, risk/urgency keywords
        │
        ▼
3. Hard gate — CRITICAL risk terms escalate immediately, skipping retrieval
        │
        ▼
4. Retrieve — semantic search over data/ via Chroma (top-k, distance-filtered)
        │
        ▼
5. Ground — draft an answer from retrieved chunks only, then verify citations
        │
        ▼
6. Company compliance check — HackerRank score-integrity / Visa PCI-DSS gate
        │
        ▼
7. Precision check — does the drafted answer actually address the ticket?
        │
        ▼
8. Final analysis pass — LLM re-verifies status/response/justification against
   the retrieved context before it's written out
        │
        ▼
output.csv
```

- **Retrieval (RAG):** Markdown docs under `data/` are chunked (`langchain-text-splitters`, header-aware + recursive character splitting) and embedded into a local Chroma collection (`chroma_db/`). Tickets are answered only from retrieved context, never from the model's own parametric knowledge — the system prompt explicitly instructs the model to ignore any instructions embedded in ticket text and stick to `<retrieved_context_documents>`.
- **Per-company routing:** `agents/` holds one agent per product ecosystem (`hackerrank_agent.py`, `claude_agent.py`, `visa_agent.py`) plus a shared base (`support_agent.py`) that handles risk/urgency assessment, company detection, and escalation logic.
- **Grounding pipeline:** `agents/rag_agent.py` (`RagAgent`) owns the actual draft → verify → precision loop and the resulting `grounded`/`precise`/`status`/`response`/`justification` state — `SupportAgent.groundness()` just delegates to it and mirrors the result back onto the agent.
- **Company-specific compliance hooks:** `SupportAgent._get_verify_hook()` (default: none) lets a subclass plug an extra check into `RagAgent`'s verification gate without `RagAgent` needing to know about any specific company. `HackerrankAgent` blocks score-manipulation promises ("increase your score", "regrade", etc.); `VisaAgent` runs an LLM-based PCI-DSS/financial-policy check. Both run only after the base citation-verification gate already passed.
- **LLM:** `models/support_agent_model.py` wraps the configured chat provider (currently Google Gemini via an OpenAI-compatible endpoint); `models/chroma_model.py` wraps the vector store (HuggingFace embeddings, local, no embedding API key needed). `models/chroma_model_claude.py` is an in-progress alternate embedding backend.
- **Prompting:** `src/prompt_builder.py` renders the final ticket + retrieved-context XML prompt; `src/ticket_analyzer.py` orchestrates the whole classify → retrieve → ground → decide flow per ticket; `src/constants.py` is the single source of truth for prompts, paths, and tunables.

## Repository layout

```
agents/
  support_agent.py       # Base agent: classify, risk/urgency, retrieval, escalation logic
  rag_agent.py            # Grounding pipeline: draft -> verify -> precision -> evaluate
  hackerrank_agent.py     # + HackerRank score-integrity compliance hook
  claude_agent.py         # Claude-scoped retrieval (no extra compliance hook)
  visa_agent.py           # + Visa PCI-DSS / financial-policy compliance hook
models/
  gemini_model.py         # Shared provider config (name/url/key/embedding model)
  support_agent_model.py  # Chat completions client + retry/rate-limit handling
  chroma_model.py         # Chroma vector store wrapper (HF embeddings)
  chroma_model_claude.py  # Alternate embedding backend (in progress)
pipelines/
  rag.py                  # Ingests data/ into the Chroma collection
  process_tickets.py      # Iterates tickets, runs TicketAnalyzer, collects output rows
src/
  constants.py            # Paths, env vars, prompts, thresholds — no magic values elsewhere
  data_handler.py         # CSV load/write via pandas
  document_handler.py     # Loads, cleans, and chunks the data/ knowledge base
  metadata_extractor.py   # Pulls company/product-area/checksum metadata from docs
  prompt_builder.py       # Builds the final per-ticket XML + instructions prompt
  ticket_analyzer.py      # Orchestrates classify -> retrieve -> ground -> respond
  enums.py                # Company, RequestType, Risk, Status, Urgency, RagStatus
  utils.py                # Timers, logging, progress bar, banner, shared helpers
support_tickets/
  support_tickets.csv          # Full input dataset
  sample_support_tickets.csv   # Small dataset for dev/testing
  output.csv                   # Agent-produced results (evaluator reads this — don't rename)
data/
  claude/                 # Claude Help Center export (knowledge base)
  hackerrank/              # HackerRank support articles (knowledge base)
  visa/                    # Visa support articles (knowledge base)
chroma_db/                # Local persisted vector store (generated)
tests/unit/                # Pytest unit tests
main.py                   # Entry point
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

| Variable          | Purpose                                                 |
|-------------------|---------------------------------------------------------|
| `MODEL_API_KEY`   | LLM provider API key                                    |
| `MODEL_API_URL`   | LLM provider base URL (optional)                        |
| `MODEL_NAME`      | Model identifier string                                 |
| `MODEL_EMBEDDING` | HuggingFace embedding model identifier                  |
| `HF_TOKEN`        | HuggingFace token (if the embedding model requires one) |
| `APP_NAME`        | Display name for the CLI banner (optional)              |

Never commit `.env` — it's already gitignored.

## Running

```bash
source venv/bin/activate
python main.py --sample --rag --refresh   # sample dataset, rebuild the vector store from data/
python main.py --sample                   # sample dataset, reuse the existing chroma_db/
python main.py --rag --refresh            # full dataset, rebuild the vector store
python main.py                            # full dataset, reuse the existing chroma_db/
```

Recognized flags (`src/constants.py::ARGS_LIST`): `--eda`, `--rag`, `--refresh`, `--sample`.

Results are written to `support_tickets/output.csv` with columns: `issue`, `subject`, `company`, `response`, `product_area`, `status`, `request_type`, `justification`.

## Testing

```bash
pytest
```

Current coverage is limited to `tests/unit/data/test_data_handler.py`; the agent/grounding pipeline is currently verified through manual runs against `log.txt`, not automated tests.

## Status & known limitations

This is a work in progress, tracked here for transparency:

- **`pipelines/process_tickets.py` currently only processes `idx == 0`** and calls `sys.exit(0)` right after, as a deliberate debug gate for iterating on a single ticket without burning tokens on the full dataset. **Both need to be removed before a real batch run** — as-is, `output.csv` is never written.
- Retrieval quality is tuned but not perfect: `CHROMA_RESULT_CNT` (top-k before the relevance-distance filter) is a real trade-off between surfacing more borderline-relevant chunks and diluting the grounding prompt with noise. Some tickets escalate because the right chunk exists in the KB but doesn't rank in the retrieved set, not because the KB genuinely lacks the answer.
- Markdown source docs contain embedded image links; these are stripped before chunking (`DocumentHandler._format_html`) so they don't eat into chunk-size budgets or dilute embeddings, but very image-heavy articles can still end up thinly covered.
- The LLM provider is Google Gemini via an OpenAI-compatible endpoint; `models/chroma_model_claude.py` is an in-progress alternate embedding backend, not yet wired into the main pipeline.
- `log.txt` captures the retrieval/grounding/verification/precision stages via `log_chat_transcript`, but the final analysis call in `TicketAnalyzer._get_output()` isn't wrapped in a log call — its result only appears if `process_tickets.py` explicitly logs `output_rows` (see the debug line near the `sys.exit(0)`).

## Key conventions

- All paths, env vars, and prompts live in `src/constants.py` — don't hardcode them elsewhere.
- `data/` is reference-only knowledge base content — don't modify it.
- `support_tickets/output.csv` and `main.py` are fixed entry points the evaluator relies on — don't rename them.
- Secrets come only from environment variables, never hardcoded.
- Company-specific business rules (score-integrity, PCI-DSS, etc.) are added by overriding `SupportAgent._get_verify_hook()` in that company's agent subclass — not by modifying `RagAgent` itself.

## License

MIT — see [LICENSE](./LICENSE).
