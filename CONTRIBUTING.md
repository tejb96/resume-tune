# Contributing

Pull requests are welcome but not expected — this is primarily a personal tool.

## Developer guide

See **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** for setup, tests, package layout, and debugging.

Quick start:

```bash
uv sync --extra dev
cp .env.example .env
cp background.example.md background.md
uv run pytest
uv run streamlit run app.py
```

## Pull request expectations

- Keep changes focused on one concern
- Add tests for behavior changes
- Match existing code style
- Do not commit `.env`, `background.md`, or `./output/`

For task-oriented edit guides, see [docs/TWEAKING.md](docs/TWEAKING.md).
