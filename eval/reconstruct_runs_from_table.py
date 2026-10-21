"""Rebuild per-run JSON rows from the per-run tables run_eval.py printed, for a
run set that was interrupted before its JSON was written (phase2_b_fewshot.txt,
runs 1-2). Only fields printed in the table are recovered; urgency and latency
were not printed, so they come back as None.

Long ticket ids overflow the fixed-width columns, so each line is parsed by
matching known values in order (ticket id and expected values from the tickets
file, then the known action / mark / category vocabularies) instead of by
column position.

Usage:
    python -m eval.reconstruct_runs_from_table eval/phase2_b_fewshot.txt out.json
"""

import json
import re
import sys
from pathlib import Path

from eval.run_eval import TICKETS_PATH

_ACTIONS = sorted(
    ["auto_resolved", "escalated", "escalated_due_to_error", "escalated_output_blocked",
     "escalated_incomplete_lookup", "http_error"],
    key=len, reverse=True,
)
_MARKS = ["XPORT-ERR", "OUT-BLOCK", "INC-LOOKUP", "HTTP-ERR", "FAIL", "OK"]


def _take(s: str, options: list[str]) -> tuple[str, str]:
    s = s.lstrip()
    for o in options:
        if s.startswith(o):
            return o, s[len(o):]
    raise ValueError(f"none of {options} at: {s[:40]!r}")


def parse(path: Path) -> list[list[dict]]:
    tickets = json.loads(TICKETS_PATH.read_text())
    by_id = {t["ticket_id"]: t for t in tickets}
    runs, current = [], None
    for line in path.read_text().splitlines():
        if re.match(r"=== Run \d+/\d+ ===", line):
            current = []
            runs.append(current)
            continue
        if current is None or len(current) == len(tickets):
            continue
        t = tickets[len(current)]
        if not line.startswith(t["ticket_id"]):
            continue
        rest = line[len(t["ticket_id"]):]
        _, rest = _take(rest, [str(t["expected_action"])])
        actual, rest = _take(rest, _ACTIONS)
        mark, rest = _take(rest, _MARKS)
        _, rest = _take(rest, [str(t["expected_category"])])
        act_cat, guard, lookup = rest.split()
        current.append({
            "ticket_id": t["ticket_id"],
            "expected_action": t["expected_action"],
            "actual_action": actual,
            "http_error": "unknown (reconstructed)" if mark == "HTTP-ERR" else None,
            "is_http_error": mark == "HTTP-ERR",
            "is_transport_error": mark == "XPORT-ERR",
            "is_output_blocked": mark == "OUT-BLOCK",
            "is_incomplete_lookup": mark == "INC-LOOKUP",
            "action_match": mark == "OK",
            "expected_category": t["expected_category"],
            "actual_category": None if act_cat == "None" else act_cat,
            "actual_urgency": None,
            "guardrail_expected": bool(t["expected_guardrail_flags"]),
            "guardrail_actual": None,
            "guardrail_match": guard == "OK",
            "expect_lookup": t.get("expect_account_lookup"),
            "actual_lookup": None,
            "lookup_match": None if lookup == "-" else lookup == "OK",
            "latency_ms": None,
        })
    complete = [r for r in runs if len(r) == len(tickets)]
    assert all(r["ticket_id"] in by_id for run in complete for r in run)
    return complete


if __name__ == "__main__":
    runs = parse(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(runs, indent=2))
    print(f"Reconstructed {len(runs)} complete run(s) into {sys.argv[2]}")
