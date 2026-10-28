"""Download ChromaDB's default embedding model archive with resume and retries.

Chroma's own downloader restarts from zero on every failure, so one flaky link can
fail a whole Docker build. This writes the same archive (same URL, same path, same
SHA-256 check that Chroma uses) and resumes partial downloads with HTTP Range
requests. Chroma then finds a valid archive and only extracts it.

Usage: python scripts/fetch_embedding_model.py
"""

import hashlib
import sys
import time

import httpx
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2 as Model

ATTEMPTS = 30


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    target = Model.DOWNLOAD_PATH / Model.ARCHIVE_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and _sha256(target) == Model._MODEL_SHA256:
        print(f"embedding model archive already present: {target}")
        return 0

    for attempt in range(1, ATTEMPTS + 1):
        have = target.stat().st_size if target.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with httpx.stream("GET", Model.MODEL_DOWNLOAD_URL, headers=headers,
                              timeout=httpx.Timeout(30.0), follow_redirects=True) as r:
                if r.status_code == 416:  # nothing left to fetch
                    pass
                elif r.status_code not in (200, 206):
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                else:
                    mode = "ab" if r.status_code == 206 else "wb"
                    with open(target, mode) as f:
                        for chunk in r.iter_bytes(1 << 16):
                            f.write(chunk)
            if _sha256(target) == Model._MODEL_SHA256:
                print(f"embedding model archive ok after {attempt} attempt(s): {target}")
                return 0
            print(f"attempt {attempt}: checksum mismatch, restarting from zero")
            target.unlink()
        except (httpx.HTTPError, OSError) as e:
            size = target.stat().st_size if target.exists() else 0
            print(f"attempt {attempt}: {type(e).__name__}, {size} bytes so far, resuming")
        time.sleep(min(2 * attempt, 15))

    print("embedding model download failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
