"""End-to-end latency benchmark for POST /ticket, to compare two server builds
(e.g. before and after the Phase 5 changes).

Sends 10 tickets from eval/test_tickets.json twice each (fresh session ids both
times), one request at a time, so the second pass shows the effect of the
classify() cache on identical messages. Reports server-side latency_ms p50/p95
overall and per pass, plus the Classifier step's latency.

Groq free-tier rate limits make the server wait for a key to refill, which would
pollute these numbers. The script reads triage_groq_rate_limited_total from
/metrics before and after and prints it: a non-zero delta means the run was
slowed by rate limiting and should not be compared.

Usage (against a running server):
    python -m eval.latency_bench --label before-phase5 --out eval/latency_before_phase5.json
"""

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

import requests

from eval.run_eval import TICKETS_PATH

SERVER_URL = os.getenv("TRIAGEFORGE_URL", "http://127.0.0.1:8000")
# Plain, low-risk tickets from the main set (no injection, no account lookup), so
# every request takes the full guardrail -> classify -> draft/escalate path.
TICKET_IDS = [
    "billing_refund", "billing_invoice_request", "technical_crash", "technical_timeout",
    "account_email_change", "account_forgot_password", "billing_update_card",
    "technical_sync_missing_notes", "account_delete", "billing_unauthorized_charges",
]


def _rate_limited_total() -> float:
    text = requests.get(f"{SERVER_URL}/metrics", timeout=10).text
    return sum(float(m) for m in re.findall(r"^triage_groq_rate_limited_total\{[^}]*\} ([0-9.e+]+)$",
                                            text, flags=re.MULTILINE))


def _pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def _summary(label: str, rows: list[dict]) -> str:
    lat = [r["latency_ms"] for r in rows]
    cls = [r["classifier_ms"] for r in rows if r["classifier_ms"] is not None]
    line = (f"  {label:<26} n={len(lat):<3} p50 {_pct(lat, 0.5):7.0f} ms   p95 {_pct(lat, 0.95):7.0f} ms"
            f"   mean {statistics.mean(lat):7.0f} ms")
    if cls:
        line += f"   classifier step p50 {_pct(cls, 0.5):6.0f} ms"
    return line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    by_id = {t["ticket_id"]: t for t in json.loads(TICKETS_PATH.read_text())}
    headers = {"X-API-Key": os.environ["TRIAGEFORGE_API_KEY"]} if os.getenv("TRIAGEFORGE_API_KEY") else {}
    rate_limited_before = _rate_limited_total()
    stamp = int(time.time())
    rows = []
    for pass_no in (1, 2):
        for tid in TICKET_IDS:
            t = by_id[tid]
            body = {"ticket_id": tid, "user_message": t["user_message"],
                    "session_id": f"latency-{args.label}-{stamp}-{pass_no}-{tid}"}
            resp = requests.post(f"{SERVER_URL}/ticket", json=body, headers=headers, timeout=3600)
            resp.raise_for_status()
            data = resp.json()
            cls = next((s.get("latency_ms") for s in data["trace"] if s["agent_name"] == "Classifier"), None)
            rows.append({"pass": pass_no, "ticket_id": tid, "final_action": data["final_action"],
                         "latency_ms": data["latency_ms"], "classifier_ms": cls})
    rate_limited = _rate_limited_total() - rate_limited_before

    lines = [
        f"Latency benchmark: {args.label}",
        f"{len(TICKET_IDS)} tickets x 2 passes, sequential, server-reported latency_ms",
        _summary("all requests", rows),
        _summary("pass 1 (first time seen)", [r for r in rows if r["pass"] == 1]),
        _summary("pass 2 (identical repeat)", [r for r in rows if r["pass"] == 2]),
        f"Groq 429s during the run: {rate_limited:.0f}"
        + ("" if rate_limited == 0 else "  <- rate-limit waits included; do not compare"),
    ]
    print("\n".join(lines))
    args.out.write_text(json.dumps({"label": args.label, "rows": rows, "rate_limited_429s": rate_limited,
                                    "summary": lines}, indent=2))


if __name__ == "__main__":
    main()
