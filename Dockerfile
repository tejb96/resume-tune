FROM python:3.13-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app.py ai.py resume.py background.md ./

RUN mkdir -p output

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD [
    "uv", "run", "streamlit", "run", "app.py",
    "--server.address", "0.0.0.0",
    "--server.port", "8501",
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false"
]
