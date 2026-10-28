# Backend: FastAPI app plus the MCP server it spawns over stdio (same image,
# separate process). The knowledge base is embedded at build time.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN chown app:app /app

USER app

# Pull ChromaDB's default embedding model (about 80 MB) before copying app code,
# so this layer is cached across code changes. The script resumes partial
# downloads (one flaky link used to fail whole builds); the second line makes
# Chroma verify and extract the archive.
COPY --chown=app:app scripts/fetch_embedding_model.py ./scripts/
RUN python scripts/fetch_embedding_model.py && \
    python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as E; E()(['check'])"

COPY --chown=app:app app ./app
COPY --chown=app:app mcp_server ./mcp_server
COPY --chown=app:app data ./data

# Builds chroma_db/, so the container starts without embedding anything.
RUN python -m app.ingest

# The cross-encoder is only used when RETRIEVAL_MODE=hybrid_rerank (not the
# default). Pass --build-arg PRELOAD_RERANKER=true to bake it in.
ARG PRELOAD_RERANKER=false
RUN if [ "$PRELOAD_RERANKER" = "true" ]; then python -m app.reranker; fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health', timeout=4)"

# PORT is set by hosts such as Render; 8000 locally.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
