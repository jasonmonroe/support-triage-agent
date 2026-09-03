@AGENTS.md

---

# Project: Support Triage Agent

HackerRank Orchestrate hackathon — AI agent that reads support tickets from a CSV and resolves them using an LLM.

## Layout

```
src/
  constants.py      # all file paths, env vars, prompts
  data_handler.py   # CSV load/write via pandas
  utils.py          # shared helpers (currently empty)
models/
  llm_model.py      # Google Gemini client wrapper
support_tickets/
  support_tickets.csv         # full dataset
  sample_support_tickets.csv  # small dataset for dev/testing
  output.csv                  # agent-produced results
data/
  claude/           # reference docs (Claude support articles)
  hackerrank/       # reference docs (HackerRank support articles)
  visa/             # reference docs
```

## Language & Runtime

- Python 3.14, virtualenv at `venv/`
- Activate before running: `source venv/bin/activate`
- Install deps: `pip install -r requirements.txt` (create if missing)

## Environment Variables

All secrets come from `.env` (never commit it). Copy `.env.example` to get started.

| Variable        | Purpose                          |
|-----------------|----------------------------------|
| `MODEL_API_KEY` | LLM provider API key             |
| `MODEL_API_URL` | LLM provider base URL (optional) |
| `MODEL_NAME`    | Model identifier string          |

Load with `python-dotenv` or `export $(cat .env)` before running.

## Key Conventions

- **All constants** live in `src/constants.py`. Do not hardcode paths, model names, or prompts inline.
- `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` in `constants.py` are the prompt surfaces — edit there, not inside the model class.
- `DataHandler` takes a `dataset` dict; pass `{"sample": True}` to use the small CSV during development.
- `LlmModel` currently uses Google Gemini (`google-genai`). Switching providers means updating `_init_model` and `generate_response` only.
- Output goes to `support_tickets/output.csv`. Do not rename this file — the evaluator reads it.

## Running the Agent

```bash
source venv/bin/activate
python main.py              # full dataset
python main.py --sample     # sample dataset (faster, for testing)
```

## Data Format

`support_tickets.csv` columns: `Issue, Subject, Company, Response, Product Area, Status, Request Type`

The agent should populate `output.csv` with its generated responses. Match the column structure expected by the evaluator (check `sample_support_tickets.csv` for reference).

## Reference Data

`data/` contains Markdown knowledge-base articles for Claude, HackerRank, and Visa products. Use these as the RAG corpus when building retrieval. Do not modify files under `data/`.

## What NOT to Do

- Never commit `.env` or hardcode API keys anywhere.
- Never rename `output.csv` or the entry-point file (`main.py`).
- Never modify files under `venv/` — update `requirements.txt` instead.
- Do not log secrets to `~/hackerrank_orchestrate/log.txt` (see AGENTS.md §2).
