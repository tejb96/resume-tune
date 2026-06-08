# Resume Tune

Local desktop-style resume generator: paste a job description, tailor summary and skills with a local OpenAI-compatible LLM (Lemonade, Ollama, etc.), preview the full resume, revise via chat, and save as DOCX or PDF.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- A running OpenAI-compatible local API (e.g. [Lemonade](https://lemonade-server.ai/) or [Ollama](https://ollama.com/))
- Optional: [LibreOffice](https://www.libreoffice.org/) for PDF export (`sudo apt install libreoffice-writer`)

## Setup

```bash
uv sync
cp .env.example .env
cp background.example.md background.md
```

Edit `.env` with your API endpoint, key, and model. For Lemonade, check your port with `lemonade status` and use `http://localhost:<port>/api/v1`. Edit `background.md` with your resume data (YAML frontmatter + career context body). `background.example.md` is the public template; `background.md` is your private copy and is gitignored.

## Run

```bash
uv run streamlit run app.py
```

## Workflow

1. Paste a job description in the sidebar and click **Generate tailored content**.
2. The app builds a DOCX immediately and shows an HTML preview on the right.
3. Use the chat on the left to request changes (e.g. *"Shorten the skills list"* or *"Emphasize Kubernetes experience"*). Each revision auto-updates the preview.
4. When satisfied, **Download DOCX** (always available) or **Download PDF** (when LibreOffice is installed). Files can also be saved to `./output`.

Advanced manual editing is available in a collapsed expander if you want to tweak summary or skills directly.

## Docker (app only)

Runs Streamlit only; point it at an LLM server on the host (or elsewhere) via `.env`:

```bash
cp .env.example .env
# For Docker → host Lemonade/Ollama, set e.g.:
# OPENAI_BASE_URL=http://host.docker.internal:<port>/api/v1

docker compose up --build
```

- App: http://localhost:8501
- Create `background.md` locally (`cp background.example.md background.md`) before starting; it is bind-mounted read-only. Generated files land in `./output`
- PDF export is not available in the Docker image unless you install LibreOffice in the container; DOCX export always works.

## Environment variables

| Variable | Description |
|----------|-------------|
| `OPENAI_BASE_URL` | OpenAI-compatible API base URL (required) |
| `OPENAI_API_KEY` | API key (`lemonade` for Lemonade; `ollama` for Ollama) |
| `OPENAI_MODEL` | Model name to use |
| `AI_OUTPUT_MAX_CHARS` | Max combined characters for tailored summary + skill labels (default `967`) |

`config.toml` supplies paths, model presets, and `ai_output_max_chars` when env vars are not set.

## Dev checks

```bash
uv run python scripts/smoke_test.py          # DOCX + HTML preview from fixtures (no LLM)
uv run python scripts/e2e_live.py            # full pipeline if API is up
uv run pytest                                # unit tests (optional: uv sync --extra dev)
```

## Project layout

| File | Role |
|------|------|
| `app.py` | Streamlit UI (preview, chat revision, save) |
| `ai.py` | Local LLM call, JSON parse/validate, revision |
| `settings.py` | `.env` + `config.toml` loader |
| `resume.py` | python-docx formatter, HTML/PDF export |
| `background.example.md` | Public resume template (frontmatter + AI context) |
| `background.md` | Your private background (gitignored; copy from example) |
| `config.toml` | Paths and model presets |
