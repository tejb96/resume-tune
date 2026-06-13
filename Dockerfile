FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/
COPY app.py config.toml background.example.md ./
RUN cp background.example.md background.md

RUN mkdir -p output

EXPOSE 8501

# HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
#     CMD uv run python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["uv","run","streamlit","run","app.py","--server.address","0.0.0.0","--server.port","8501","--server.headless","true","--browser.gatherUsageStats","false"]
