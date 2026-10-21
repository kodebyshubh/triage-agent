"""CPU benchmark for the cross-encoder re-ranker: sentence-transformers (torch)
vs the same model's ONNX export through onnxruntime (what app/reranker.py uses).

Each backend runs in its own fresh process so resident memory is measured
cleanly. Scores 10 (query, chunk) pairs, which matches RERANK_CANDIDATES.

Usage (torch is not a project dependency, so pass an interpreter that has it):
    python -m eval.reranker_cpu_benchmark --torch-python path/to/python
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_WORKER = r'''
import json, os, statistics, sys, time
import psutil
proc = psutil.Process()
base = proc.memory_info().rss
backend = sys.argv[1]
pairs = json.loads(sys.argv[2])
t0 = time.perf_counter()
if backend == "sentence-transformers":
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")
    score = lambda: [float(s) for s in model.predict(pairs)]
else:
    sys.path.insert(0, sys.argv[3])
    from app.reranker import rerank
    q = pairs[0][0]
    score = lambda: rerank(q, [d for _, d in pairs])
scores = score()
load_ms = (time.perf_counter() - t0) * 1000
times = []
for _ in range(30):
    t = time.perf_counter(); score(); times.append((time.perf_counter() - t) * 1000)
times.sort()
print(json.dumps({"load_ms": load_ms, "p50_ms": statistics.median(times),
                  "p95_ms": times[int(len(times) * 0.95)],
                  "rss_mb": proc.memory_info().rss / 2**20,
                  "rss_added_mb": (proc.memory_info().rss - base) / 2**20,
                  "scores": scores}))
'''


def _pairs() -> list[list[str]]:
    sys.path.insert(0, str(ROOT))
    from app.retrieval import rank

    query = "I want my money back for something I bought two weeks ago"
    return [[query, h.document] for h in rank(query, "vector")[:10]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-python", required=True)
    args = parser.parse_args()

    pairs = json.dumps(_pairs())
    results = {}
    for backend, python in (("onnxruntime", sys.executable), ("sentence-transformers", args.torch_python)):
        out = subprocess.run(
            [python, "-c", _WORKER, backend, pairs, str(ROOT)],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[-1]
        results[backend] = json.loads(out)

    print("Cross-encoder ms-marco-MiniLM-L6-v2, 10 (query, chunk) pairs, CPU, 30 timed runs\n")
    print(f"{'backend':<24}{'load ms':>9}{'p50 ms':>9}{'p95 ms':>9}{'RSS MB':>9}{'added MB':>10}")
    for backend, r in results.items():
        print(
            f"{backend:<24}{r['load_ms']:>9.0f}{r['p50_ms']:>9.1f}{r['p95_ms']:>9.1f}"
            f"{r['rss_mb']:>9.0f}{r['rss_added_mb']:>10.0f}"
        )
    a, b = results["onnxruntime"]["scores"], results["sentence-transformers"]["scores"]
    print(f"\nMax abs score difference between backends: {max(abs(x - y) for x, y in zip(a, b)):.5f}")
    same_order = sorted(range(10), key=lambda i: -a[i]) == sorted(range(10), key=lambda i: -b[i])
    print(f"Same ranking order: {same_order}")


if __name__ == "__main__":
    main()
