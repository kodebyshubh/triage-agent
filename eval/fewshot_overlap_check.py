"""Checks that no few-shot example in the classify() prompt is copied from, or a
near-duplicate of, any eval ticket. No LLM calls.

Three similarities per (example, ticket) pair:
- char ratio: difflib.SequenceMatcher on lowercased text
- word Jaccard: overlap of lowercased word sets, stopwords removed
- embedding cosine: ChromaDB's default local embedding model (all-MiniLM-L6-v2),
  the same one the KB uses

A pair counts as a near-duplicate if ANY of these is at or above its threshold.
The thresholds were fixed before the check was first run.

Usage:
    python -m eval.fewshot_overlap_check [extra_tickets.json ...]
"""

import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from app.agents.classifier import _SYSTEM_PROMPT
from eval.run_eval import TICKETS_PATH

CHAR_RATIO_MAX = 0.60
JACCARD_MAX = 0.50
COSINE_MAX = 0.80

_STOP = set(
    "a an and are as at be but by can do for from have how i if in is it its me my "
    "of on or so that the this to was what when where which why will with you your".split()
)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP}


def main() -> None:
    examples = re.findall(r'^Message: "(.+)"$', _SYSTEM_PROMPT, flags=re.MULTILINE)
    files = [TICKETS_PATH] + [Path(p) for p in sys.argv[1:]]
    tickets = [(f.name, t) for f in files for t in json.loads(f.read_text())]

    embed = DefaultEmbeddingFunction()
    ex_vecs = np.array(embed(examples))
    tk_vecs = np.array(embed([t["user_message"] for _, t in tickets]))
    ex_vecs /= np.linalg.norm(ex_vecs, axis=1, keepdims=True)
    tk_vecs /= np.linalg.norm(tk_vecs, axis=1, keepdims=True)
    cos = ex_vecs @ tk_vecs.T

    print(f"{len(examples)} few-shot examples vs {len(tickets)} tickets "
          f"({', '.join(f.name for f in files)})")
    print(f"Near-duplicate if char ratio >= {CHAR_RATIO_MAX}, word Jaccard >= {JACCARD_MAX}, "
          f"or embedding cosine >= {COSINE_MAX}\n")

    flagged = 0
    for i, ex in enumerate(examples):
        best = {"char": (0.0, ""), "jac": (0.0, ""), "cos": (0.0, "")}
        for j, (_, t) in enumerate(tickets):
            msg = t["user_message"]
            char = difflib.SequenceMatcher(None, ex.lower(), msg.lower()).ratio()
            a, b = _words(ex), _words(msg)
            jac = len(a & b) / len(a | b) if a | b else 0.0
            for key, val in (("char", char), ("jac", jac), ("cos", float(cos[i, j]))):
                if val > best[key][0]:
                    best[key] = (val, t["ticket_id"])
        dup = (best["char"][0] >= CHAR_RATIO_MAX or best["jac"][0] >= JACCARD_MAX
               or best["cos"][0] >= COSINE_MAX)
        flagged += dup
        print(f"[{'NEAR-DUPLICATE' if dup else 'ok'}] {ex}")
        print(f"    max char ratio {best['char'][0]:.2f} ({best['char'][1]})")
        print(f"    max word Jaccard {best['jac'][0]:.2f} ({best['jac'][1]})")
        print(f"    max embedding cosine {best['cos'][0]:.2f} ({best['cos'][1]})")
    print(f"\nResult: {flagged}/{len(examples)} examples flagged as near-duplicates.")


if __name__ == "__main__":
    main()
