# Development

Notes for running and hacking on Resume Tune locally.

## Setup

```bash
git clone https://github.com/tejb96/resume-tune.git
cd resume-tune
uv sync --extra dev
cp .env.example .env
cp background.example.md background.md
```

Edit `.env` for your LLM endpoint. Edit `background.md` with test data (never commit your real resume).

## Run the app

```bash
uv run streamlit run app.py
```

## Tests

```bash
uv run pytest
uv run python scripts/smoke_test.py --static-only
uv run python scripts/ats_check.py --jd-file path/to/jd.txt
```

The smoke test builds a DOCX from fixtures without calling an LLM. Use `uv run python scripts/e2e_live.py` only when a local API is running. `ats_check.py` prints a deterministic ATS report as JSON (no LLM).

## Code areas

| Module | Responsibility |
|--------|----------------|
| `app.py` | Streamlit UI |
| `ai.py` | LLM calls, JSON parsing, chat revision |
| `selection.py` | Job-aware bullet scoring and selection |
| `resume.py` | DOCX/HTML/PDF export and page fitting |
| `ats.py` | Deterministic ATS compatibility checks |
| `settings.py` | `.env` and `config.toml` loading |

## Pull requests

Keep changes focused, add tests for behavior changes, and match existing style. Do not commit `.env`, `background.md`, or `./output/`. Pull requests are welcome but not expected — this is primarily a personal tool.
