# Resume Tune

![views](https://raw.githubusercontent.com/tejb96/resume-tune/traffic-data/views-badge.svg)
![clones](https://raw.githubusercontent.com/tejb96/resume-tune/traffic-data/clones-badge.svg)

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

Edit `.env` with your API endpoint, key, and model in the **Settings** page (or copy from `.env.example`). Edit `background.md` on the **Resume data** page or from [`background.example.md`](background.example.md) — see [docs/BACKGROUND.md](docs/BACKGROUND.md). `background.md` is gitignored.

## Run

```bash
uv run streamlit run Tailor_Resume.py
```

## Workflow

Use the sidebar to switch between **Tailor resume**, **Resume data**, and **Settings**.

### Background preview (no API)

1. Click **Preview from background** in the sidebar.
2. Static sections from `background.md` render without summary/skills.
3. Edit resume data on the **Resume data** page and preview again.

```bash
uv run python scripts/smoke_test.py --background background.md --static-only
```

### Job-tailored generate

1. Paste a job description and click **Generate tailored content**.
2. DOCX builds immediately; HTML preview appears on the right.
3. Use chat to revise (e.g. *"Shorten the skills list"*).
4. **Download DOCX** or **Download PDF** (LibreOffice required). Files can save to `./output`.

Manual summary/skills editing is available in a collapsed expander.

### ATS check and tracker

- **ATS check** — After generating, run **Run ATS check** in the sidebar. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md).
- **Application tracker** — Optional form logs applications to `./output/applications.xlsx`.

CLI: `uv run python scripts/ats_check.py --jd-file path/to/jd.txt --background background.md`

## Docker

```bash
cp .env.example .env
docker compose up --build
```

App: http://localhost:8501. Create `background.md` locally before starting (or use **Resume data** → **Create from example** in the app); it is bind-mounted read-write. PDF export needs LibreOffice in the container.

## Documentation

Full guides for configuration, architecture, and customization:

**[docs/README.md](docs/README.md)** — start here (1-hour onboarding path)

| Doc | Contents |
|-----|----------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline, package layout, data structures |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | `config.toml` and `.env` reference |
| [BACKGROUND.md](docs/BACKGROUND.md) | `background.md` schema |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | Tests, scripts, debugging |
| [TWEAKING.md](docs/TWEAKING.md) | Customization cookbook |

## Dev checks

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/smoke_test.py --static-only
uv run python scripts/e2e_live.py    # requires running LLM API
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Project layout

```
Tailor_Resume.py       Streamlit entry (Tailor resume)
pages/                 Resume data and Settings pages
config.toml            Settings
background.example.md  Resume template
src/resume_tune/       Python package (llm, render, skills, ats, …)
docs/                  Developer documentation
scripts/               CLI smoke tests
tests/                 pytest suite
```

## License

MIT — see [LICENSE](LICENSE).
