# Configuration reference

Settings are loaded by [`src/resume_tune/settings.py`](../src/resume_tune/settings.py) from the repo root.

## Precedence

```
.env environment variables  →  config.local.toml  →  config.toml  →  code defaults
```

Edit settings in the app (**Settings** page in the sidebar) or by hand:

- **LLM credentials** — saved to `.env` (endpoint, API key, model, `AI_OUTPUT_MAX_CHARS`)
- **Other options** — saved to `config.local.toml` (gitignored overlay over committed `config.toml`)

The Settings page clears the Streamlit config cache on save so changes apply on the next rerun without restarting the app. Hand-editing `config.toml` still requires a Streamlit restart (or navigating away and back).

## Environment variables (`.env`)

Copy from [`.env.example`](../.env.example).

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_BASE_URL` | OpenAI-compatible API base URL | `http://localhost:8000/api/v1` |
| `OPENAI_API_KEY` | API key | `lemonade` or `ollama` |
| `OPENAI_MODEL` | Model name | `llama3.1:8b-FLM` |
| `AI_OUTPUT_MAX_CHARS` | Max combined chars for summary + skill labels | `600` |

For Lemonade, check your port with `lemonade status`. For Docker → host LLM, use `http://host.docker.internal:<port>/api/v1`.

Legacy alias: `OPENAI_API_BASE` is accepted instead of `OPENAI_BASE_URL`.

## `config.toml` keys

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `endpoint_url` | string | `""` | LLM base URL when `OPENAI_BASE_URL` unset |
| `api_key` | string | `"lemonade"` | API key when `OPENAI_API_KEY` unset |
| `model_name` | string | `"llama3.1:8b-FLM"` | Model when `OPENAI_MODEL` unset |
| `output_dir` | string | `"./output"` | Where saved DOCX/PDF files go |
| `background_file` | string | `"./background.md"` | Path to your resume data |
| `tracker_file` | string | `"./output/applications.xlsx"` | Application log spreadsheet |
| `ai_output_max_chars` | int | `600` | Character budget for AI summary + skills |
| `max_resume_pages` | int | `1` | Target page count (requires LibreOffice for PDF trial render) |
| `auto_fill_page_budget` | bool | `true` | Greedily add highest-scored omitted bullets to fill pages |
| `overflow_warning_min_composite` | float | `75.0` | Score threshold (75 = rating 4+) for overflow warnings |
| `enable_job_aware_selection` | bool | `true` | LLM rates bullets 1–5 per job; enables smart selection |
| `enable_ats_compat` | bool | `true` | Show ATS check UI in sidebar |
| `min_project_entries` | int | `1` | Minimum projects kept during page fit |
| `min_project_bullets` | int | `1` | Minimum bullets per kept project |
| `max_skill_categories` | int | `3` | Max skill lines in export |
| `max_skills_per_category` | int | `5` | Max skills per line |
| `max_chars_per_skill_line` | int | `88` | Character limit per skill line |
| `max_completion_tokens` | int | auto | Optional hard cap on LLM completion tokens |
| `max_certifications` | int | `1` | Export only first N certifications from YAML |
| `resume_sections` | list | see below | Sections and render order |

### `ai_output_max_chars` note

The code fallback in `llm/ai.py` is `967` (a reference fit for a two-page template). At runtime, **`config.toml` default `600` wins** unless you set `AI_OUTPUT_MAX_CHARS` in `.env`. Tune this if summary/skills overflow or feel too sparse.

### `resume_sections`

Valid ids: `summary`, `skills`, `experience`, `education`, `projects`, `certifications`. Header is always first.

```toml
resume_sections = [
    "skills",
    "experience",
    "education",
    "projects",
    "certifications",
]
```

- Omit a section to exclude it (e.g. drop `"summary"` for no professional summary)
- Reorder entries to rearrange sections
- Omitting `summary` and/or `skills` skips LLM generation for those fields
- If both are omitted, the app builds from `background.md` only — no JD or LLM required

Background-backed sections render only when their YAML lists are non-empty.

## Page fitting and job-aware selection

When `enable_job_aware_selection = true`:

1. LLM rates each experience/project/education item 1–5 for the job
2. App builds an initial quality-ranked selection
3. **Auto-fill** adds highest-scored omitted bullets until the page budget is full (trial PDF render)
4. If still over `max_resume_pages`, the fit loop trims lowest-scored bullets

Skills are optimized at generation time and are **not** modified during page fitting.

If enough high-scoring content remains for a full extra page, the app shows an overflow notice suggesting a higher `max_resume_pages`. Use **Page fit details** in the sidebar for expand/trim logs.

Keep a rich `background.md` as source of truth; the app selects what ships per job rather than rewriting bullets. List your primary certification first when `max_certifications = 1`.

## Model presets

The `[models]` table in `config.toml` is informational for the UI — it does not auto-select models:

```toml
[models]
ollama = ["llama3.2", "mistral", "qwen2.5"]
lemonade = ["llama3.1:8b", "default"]
```

Set the active model via `OPENAI_MODEL` in `.env` or the **Settings** page.

## In-app editors

| Page | What it edits |
|------|----------------|
| **Tailor resume** | Job description, preview, AI generation, export |
| **Resume data** | `background.md` — structured resume sections, career narrative, or raw file |
| **Settings** | `.env` LLM credentials and `config.local.toml` overrides |

Saving `background.md` from the app writes a `.bak` backup beside the file.
