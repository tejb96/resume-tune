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

### Background preview (no API)

1. Click **Preview from background** in the sidebar — no job description or LLM endpoint required.
2. The app renders static sections from `background.md` (experience, education, projects, certifications). Summary and skills are omitted so you can check bullet wording and layout.
3. Edit `background.md`, click **Preview from background** again to refresh.

CLI equivalent:

```bash
uv run python scripts/smoke_test.py --background background.md --static-only
```

### Job-tailored generate

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

### Resume section layout

`resume_sections` in `config.toml` controls which sections appear in the exported DOCX/PDF and their order (the header is always first). Valid section ids: `summary`, `skills`, `experience`, `education`, `projects`, `certifications`.

```toml
resume_sections = [
    "summary",
    "skills",
    "experience",
    "education",
    "projects",
    "certifications",
]
```

- Omit a section to exclude it (e.g. remove `"summary"` to drop the professional summary).
- Reorder entries to rearrange sections (e.g. put `"certifications"` before `"education"`).

Omitting `summary` and/or `skills` also skips LLM generation for those fields. The character budget (`ai_output_max_chars`) applies only to included AI sections. If both are omitted, the app builds the resume from `background.md` only — no job description or LLM endpoint required.

Background-backed sections still render only when their YAML lists are non-empty. Restart Streamlit after editing `config.toml` so cached settings reload.

### One-page layout and job-aware selection

`config.toml` controls PDF page fitting (requires LibreOffice), job-aware bullet selection, and skills layout:

```toml
max_resume_pages = 1
auto_fill_page_budget = true
overflow_warning_min_composite = 75
enable_job_aware_selection = true
min_project_entries = 1
min_project_bullets = 1
max_skill_categories = 4
max_skills_per_category = 5
max_chars_per_skill_line = 88
max_certifications = 1
```

When `enable_job_aware_selection` is true, the LLM rates each experience/project/education item 1–5 for the job. The app builds an initial quality-ranked selection, then **auto-fills** the page budget by adding the highest-scored omitted bullets (trial-rendered via PDF so a partial second page never appears). If content still exceeds `max_resume_pages`, the fit loop trims lowest-scored bullets. Skills are optimized at generation time and are **not** modified during page fitting.

If enough high-scoring content remains to fill a full additional page, the app shows an overflow notice suggesting a higher `max_resume_pages`. Use **Page fit details** in the sidebar to see counts, expand/trim logs, and warnings.

Keep a rich `background.md` as source of truth; the app selects what ships per job rather than rewriting bullets. List your primary certification first in YAML when using `max_certifications = 1` (e.g. AWS SAA).

## Dev checks

```bash
uv run python scripts/smoke_test.py          # DOCX + HTML preview from fixtures (no LLM)
uv run python scripts/smoke_test.py --background background.md --static-only
uv run python scripts/e2e_live.py            # full pipeline if API is up
uv run pytest                                # unit tests (optional: uv sync --extra dev)
```

## Project layout

| File | Role |
|------|------|
| `app.py` | Streamlit UI (preview, chat revision, save) |
| `ai.py` | Local LLM call, JSON parse/validate, revision |
| `selection.py` | Job-aware experience/project bullet selection by index |
| `settings.py` | `.env` + `config.toml` loader |
| `resume.py` | python-docx formatter, HTML/PDF export |
| `background.example.md` | Public resume template (frontmatter + AI context) |
| `background.md` | Your private background (gitignored; copy from example) |
| `config.toml` | Paths and model presets |
