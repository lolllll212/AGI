# syntax=docker/dockerfile:1

# 1. Build the web console.
FROM node:22-alpine AS web
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN mkdir -p /app/src/agi/infrastructure/api && npm run build

# 2. The server.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY --from=web /app/src/agi/infrastructure/api/static/ src/agi/infrastructure/api/static/
RUN pip install ".[graph]"
COPY config/ config/
COPY plugins/ plugins/
RUN useradd --create-home --uid 10001 agi && mkdir -p /data/workspace /data/state && chown -R agi:agi /data
USER agi
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
ENTRYPOINT ["agi"]
CMD ["serve", "--config", "config/docker.toml"]
