# Development guide

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

`uv sync` installs the `resume_tune` package in editable mode from `src/`. Scripts and tests import from `resume_tune.*` — no `sys.path` hacks needed.

## Run the app

```bash
uv run streamlit run Tailor_Resume.py
```

Clear Streamlit cache after config changes: sidebar menu → **Clear cache**, or restart the process.

## Package layout

```
src/resume_tune/
├── settings.py          # .env + config.toml
├── llm/                 # LLM calls (ai.py, selection.py)
├── content/             # Pure scoring policy (scoring.py)
├── skills/              # Skill map + packing
├── render/              # DOCX/HTML/PDF + page fit (resume.py)
├── ats/                 # ATS checks
└── tracker/             # Application spreadsheet
Tailor_Resume.py         # Streamlit entry — Tailor resume (repo root)
pages/                   # Resume data and Settings
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for data flow and key dict shapes.

## Test matrix

| Command | Needs LLM? | Purpose |
|---------|------------|---------|
| `uv run pytest` | No | Unit tests (137+) |
| `uv run python scripts/smoke_test.py --static-only` | No | DOCX/HTML from fixtures |
| `uv run python scripts/smoke_test.py` | No | Full design preview with sample AI output |
| `uv run python scripts/ats_check.py --jd-file jd.txt` | No | ATS JSON report |
| `uv run python scripts/e2e_live.py` | Yes | Full pipeline smoke (skips if API down) |

Pytest `pythonpath` includes both `src` and `.` so tests can import `resume_tune` and root `Tailor_Resume.py`.

When mocking in tests, patch where the symbol is **used**:

```python
patch("resume_tune.render.resume.docx_to_pdf", ...)
patch("resume_tune.llm.ai.OpenAI", ...)
```

## Debugging

### LLM JSON parse failures

- Check `llm/ai.py` — retries and fence stripping in `strip_json_fences`
- Run with a smaller model or lower `max_completion_tokens`
- Inspect raw model output in exception messages (`AIResponseError`)

### Skills dropped unexpectedly

- Skill must exist in `skills_map` with exact spelling
- Check guardrails in `llm/ai.py` (`apply_skills_guardrails`)
- Run `uv run pytest tests/test_skills_map.py tests/test_skills_selection.py`

### Page fit trimming too aggressively

- Increase `max_resume_pages` in `config.toml`
- Disable `auto_fill_page_budget` to stop greedy expansion
- Check **Page fit details** diagnostics or `artifacts["diagnostics"]` from `build_resume_artifacts()`
- See [TWEAKING.md](TWEAKING.md) → page fit section

### ATS false positives/negatives

- ATS is deterministic keyword matching — see `ats/ats.py` (`TECH_TERMS`, section detection)
- Run CLI report: `uv run python scripts/ats_check.py --jd-file jd.txt --background background.md`

### Import errors after editing layout

Always run `uv sync` after changing `pyproject.toml`. Imports use `from resume_tune...` — flat `from ai import` no longer works.

## Docker

```bash
docker compose up --build
```

The image copies `src/`, `pages/`, `Tailor_Resume.py`, and config assets. Mount `background.md` and `./output` via `docker-compose.yml`. PDF export requires LibreOffice in the container (not installed by default).

## Pull requests

Keep changes focused, add tests for behavior changes, and match existing style. Do not commit `.env`, `background.md`, or `./output/`.

See [TWEAKING.md](TWEAKING.md) for task-oriented edit guides.
