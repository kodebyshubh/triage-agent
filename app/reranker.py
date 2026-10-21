"""Cross-encoder re-ranker: cross-encoder/ms-marco-MiniLM-L6-v2 run through
onnxruntime instead of sentence-transformers.

Same model weights (the ONNX export published in the model's own Hugging Face
repo), but it needs no torch: onnxruntime and tokenizers are already installed
as ChromaDB dependencies. See eval/reranker_cpu_benchmark.txt for the CPU latency
and memory comparison behind this choice.

Model files are downloaded once into models/ (gitignored) on first use, or ahead
of time with `python -m app.reranker`.
"""

from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np

MODEL_REPO = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "ms-marco-MiniLM-L6-v2"
_FILES = {"onnx/model.onnx": "model.onnx", "tokenizer.json": "tokenizer.json"}
_MAX_LENGTH = 512


def download_model() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for remote, local in _FILES.items():
        target = MODEL_DIR / local
        if target.exists():
            continue
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{remote}"
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            tmp = target.with_suffix(".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
            tmp.replace(target)
    return MODEL_DIR


@lru_cache(maxsize=1)
def _load():
    import onnxruntime as ort
    from tokenizers import Tokenizer

    download_model()
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=_MAX_LENGTH)
    tokenizer.enable_padding()
    session = ort.InferenceSession(
        str(MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    return tokenizer, session, {i.name for i in session.get_inputs()}


def rerank(query: str, documents: list[str]) -> list[float]:
    """Relevance logit per document (higher is more relevant)."""
    if not documents:
        return []
    tokenizer, session, input_names = _load()
    enc = tokenizer.encode_batch([(query, d) for d in documents])
    feed = {
        "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
    }
    logits = session.run(None, {k: v for k, v in feed.items() if k in input_names})[0]
    return logits[:, 0].astype(float).tolist()


if __name__ == "__main__":
    print(f"Model files in {download_model()}")
