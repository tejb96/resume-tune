# Resume Tune

Local desktop-style resume generator: paste a job description, tailor summary and skills with a local LLM (Ollama or Lemonade), and download a formatted DOCX.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- A running OpenAI-compatible local API (e.g. [Ollama](https://ollama.com/) at `http://localhost:11434/v1`)

## Setup

```bash
uv sync
```

Edit `config.toml` for your endpoint, model name, and paths. Edit `background.md` with your resume data (YAML frontmatter + career context body).

## Run

```bash
uv run streamlit run app.py
```

## Docker (app + Ollama)

Starts Streamlit and Ollama, pulls `tinyllama` (smallest practical test model) on first run:

```bash
docker compose up --build
```

- App: http://localhost:8501
- Ollama API: http://localhost:11434

`config.docker.toml` is mounted into the app container (endpoint `http://ollama:11434/v1`). Edit `background.md` locally — it is bind-mounted read-only. Generated DOCX files land in `./output`.

First startup may take a few minutes while the model downloads.

## Dev checks

```bash
uv run python scripts/smoke_test.py          # DOCX from fixtures (no LLM)
uv run python scripts/e2e_live.py            # full pipeline if Ollama/Lemonade is up
```

## Project layout

| File | Role |
|------|------|
| `app.py` | Streamlit UI |
| `ai.py` | Local LLM call, JSON parse/validate |
| `resume.py` | python-docx formatter |
| `background.md` | Master background (frontmatter + AI context) |
| `config.toml` | Endpoint, model, paths |
