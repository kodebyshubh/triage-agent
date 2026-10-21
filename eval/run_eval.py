"""POSTs eval/test_tickets.json to a live TriageForge server and prints a summary
table: expected vs actual routing, and whether guardrails triggered as expected.

Each ticket runs --runs times (default 3). Every run gets fresh session ids
(`<session_id>-r<n>`) so session memory from one run never leaks into the next.
Output: the original per-run table for each run (metrics computed exactly as
before), then an aggregate section with mean/spread per metric and per-failure-
mode rates.

Usage: start the server (uvicorn app.main:app), then run:
    python -m eval.run_eval [--runs 3] [--workers 4] [--json out.json]
"""

import argparse
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

TICKETS_PATH = Path(__file__).parent / "test_tickets.json"
SERVER_URL = os.getenv("TRIAGEFORGE_URL", "http://127.0.0.1:8000")
ORIGINAL_TICKET_COUNT = 16
# Generous by default: under Groq free-tier daily limits the server may wait for
# a rate-limited key to refill before answering.
REQUEST_TIMEOUT_S = float(os.getenv("TRIAGEFORGE_EVAL_TIMEOUT_S", "3600"))


def _actual_category(trace: list[dict]) -> str | None:
    for step in trace:
        if step["agent_name"] == "Classifier":
            return step["output"].get("category")
    return None


def _actual_urgency(trace: list[dict]) -> int | None:
    for step in trace:
        if step["agent_name"] == "Classifier":
            return step["output"].get("urgency")
    return None


def _account_lookup_called(trace: list[dict]) -> bool:
    return any(step["agent_name"] == "AccountLookup" for step in trace)


def _guardrail_triggered(result: dict) -> bool:
    if result["final_action"] == "escalated" and "Flags:" in result["response"].get(
        "flagged_reason", ""
    ):
        return True
    for step in result["trace"]:
        if step["agent_name"] == "Classifier" and "[REDACTED_" in step["input"].get(
            "user_message", ""
        ):
            return True
    return False


def _post(t: dict, run: int) -> tuple[dict | None, str | None]:
    payload = {
        "ticket_id": t["ticket_id"],
        "user_message": t["user_message"],
        "session_id": f"{t['session_id']}-r{run}",
    }
    headers = {}
    if os.getenv("TRIAGEFORGE_API_KEY"):
        headers["X-API-Key"] = os.environ["TRIAGEFORGE_API_KEY"]
    try:
        resp = requests.post(
            f"{SERVER_URL}/ticket", json=payload, headers=headers, timeout=REQUEST_TIMEOUT_S
        )
    except requests.RequestException as e:
        return None, f"{type(e).__name__}"
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"
    return resp.json(), None


def _row(t: dict, result: dict | None, http_error: str | None) -> dict:
    expected_action = t["expected_action"]
    guardrail_expected = bool(t["expected_guardrail_flags"])
    expect_lookup = t.get("expect_account_lookup")

    if result is None:
        # The request itself failed (e.g. a 500 from an upstream rate limit). Not a
        # routing decision and not any of the four final_action outcomes below, so
        # it gets its own bucket and counts as a fail for every per-ticket check.
        return {
            "ticket_id": t["ticket_id"],
            "expected_action": expected_action,
            "actual_action": "http_error",
            "http_error": http_error,
            "is_http_error": True,
            "is_transport_error": False,
            "is_output_blocked": False,
            "is_incomplete_lookup": False,
            "action_match": False,
            "expected_category": t["expected_category"],
            "actual_category": None,
            "actual_urgency": None,
            "guardrail_expected": guardrail_expected,
            "guardrail_actual": None,
            "guardrail_match": False,
            "expect_lookup": expect_lookup,
            "actual_lookup": None,
            "lookup_match": None if expect_lookup is None else False,
            "latency_ms": None,
        }

    actual_action = result["final_action"]
    # Four distinct reasons a ticket can end up not-auto-resolved. None of them
    # are a routing decision, so none count toward action_match either way -
    # conflating any of them with "escalated" or with each other would hide
    # which of routing-bias / transport-flakiness / output-guardrail-flakiness /
    # incomplete-lookup is actually responsible for a given ticket's outcome.
    is_transport_error = actual_action == "escalated_due_to_error"
    is_output_blocked = actual_action == "escalated_output_blocked"
    is_incomplete_lookup = actual_action == "escalated_incomplete_lookup"
    is_non_routing_outcome = is_transport_error or is_output_blocked or is_incomplete_lookup
    guardrail_actual = _guardrail_triggered(result)

    # Additional metric (Prompt 3): only applies to tickets that declare
    # expect_account_lookup - the original 14 tickets don't have this key,
    # so lookup_match stays None for them (not applicable), never counted
    # as a pass or a fail, and never touches the 4 existing metrics above.
    actual_lookup = _account_lookup_called(result["trace"])
    lookup_match = None if expect_lookup is None else (expect_lookup == actual_lookup)

    return {
        "ticket_id": t["ticket_id"],
        "expected_action": expected_action,
        "actual_action": actual_action,
        "http_error": None,
        "is_http_error": False,
        "is_transport_error": is_transport_error,
        "is_output_blocked": is_output_blocked,
        "is_incomplete_lookup": is_incomplete_lookup,
        "action_match": (not is_non_routing_outcome) and expected_action == actual_action,
        "expected_category": t["expected_category"],
        "actual_category": _actual_category(result["trace"]),
        "actual_urgency": _actual_urgency(result["trace"]),
        "guardrail_expected": guardrail_expected,
        "guardrail_actual": guardrail_actual,
        "guardrail_match": guardrail_expected == guardrail_actual,
        "expect_lookup": expect_lookup,
        "actual_lookup": actual_lookup,
        "lookup_match": lookup_match,
        "latency_ms": result.get("latency_ms"),
    }


def run_eval(tickets: list[dict], run: int = 1, workers: int = 4) -> list[dict]:
    def _one(t):
        result, err = _post(t, run)
        return _row(t, result, err)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, tickets))


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    lookup_rows = [r for r in rows if r["lookup_match"] is not None]
    return {
        "routing_match": sum(r["action_match"] for r in rows) / total,
        "guardrail_match": sum(r["guardrail_match"] for r in rows) / total,
        "transport_error": sum(r["is_transport_error"] for r in rows) / total,
        "output_blocked": sum(r["is_output_blocked"] for r in rows) / total,
        "incomplete_lookup": sum(r["is_incomplete_lookup"] for r in rows) / total,
        "http_error": sum(r["is_http_error"] for r in rows) / total,
        "account_lookup_match": (
            sum(r["lookup_match"] for r in lookup_rows) / len(lookup_rows) if lookup_rows else None
        ),
        "category_match": sum(r["expected_category"] == r["actual_category"] for r in rows) / total,
    }


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'ticket_id':<28}{'exp_action':<13}{'act_action':<13}{'action_ok':<10}"
        f"{'exp_cat':<10}{'act_cat':<10}{'guardrail_ok':<13}{'lookup_ok':<10}"
    )
    print(header)
    print("-" * len(header))

    n_action_match = 0
    n_guardrail_match = 0
    n_transport_error = 0
    n_output_blocked = 0
    n_incomplete_lookup = 0
    n_http_error = 0
    n_lookup_applicable = 0
    n_lookup_match = 0
    for r in rows:
        if r["is_http_error"]:
            mark = "HTTP-ERR"
        elif r["is_transport_error"]:
            mark = "XPORT-ERR"
        elif r["is_output_blocked"]:
            mark = "OUT-BLOCK"
        elif r["is_incomplete_lookup"]:
            mark = "INC-LOOKUP"
        else:
            mark = "OK" if r["action_match"] else "FAIL"
        lookup_mark = "-" if r["lookup_match"] is None else ("OK" if r["lookup_match"] else "FAIL")
        print(
            f"{r['ticket_id']:<28}{str(r['expected_action']):<13}"
            f"{str(r['actual_action']):<13}{mark:<11}"
            f"{str(r['expected_category']):<10}{str(r['actual_category']):<10}"
            f"{('OK' if r['guardrail_match'] else 'FAIL'):<13}{lookup_mark:<10}"
        )
        n_action_match += r["action_match"]
        n_guardrail_match += r["guardrail_match"]
        n_transport_error += r["is_transport_error"]
        n_output_blocked += r["is_output_blocked"]
        n_incomplete_lookup += r["is_incomplete_lookup"]
        n_http_error += r["is_http_error"]
        if r["lookup_match"] is not None:
            n_lookup_applicable += 1
            n_lookup_match += r["lookup_match"]

    total = len(rows)
    print("-" * len(header))
    print(f"Routing match:        {n_action_match}/{total} ({n_action_match / total:.0%})")
    print(f"Guardrail match:      {n_guardrail_match}/{total} ({n_guardrail_match / total:.0%})")
    print(f"Transport errors:     {n_transport_error}/{total} (excluded from routing match)")
    print(f"Output-blocked:       {n_output_blocked}/{total} (excluded from routing match)")
    print(f"Incomplete lookup:    {n_incomplete_lookup}/{total} (excluded from routing match)")
    print(f"HTTP errors:          {n_http_error}/{total} (request failed, excluded from routing match)")
    if n_lookup_applicable:
        print(
            f"Account lookup match: {n_lookup_match}/{n_lookup_applicable} "
            f"({n_lookup_match / n_lookup_applicable:.0%}) "
            "(only tickets declaring expect_account_lookup; additive metric, "
            "doesn't affect routing/guardrail match above)"
        )


_ABBREV = {
    "auto_resolved": "auto",
    "escalated": "esc",
    "escalated_due_to_error": "xport",
    "escalated_output_blocked": "blocked",
    "escalated_incomplete_lookup": "inc-lookup",
    "http_error": "http-err",
}


def _spread(values: list[float]) -> str:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"mean {mean:6.1%}  sd {sd:5.1%}  min {min(values):6.1%}  max {max(values):6.1%}"


def print_aggregate(runs: list[list[dict]]) -> None:
    n_runs = len(runs)
    print(f"\n=== Aggregate over {n_runs} run(s) ===\n")

    header = f"{'ticket_id':<34}{'exp':<6}{'actual per run':<30}{'urgency':<12}{'ok':<6}"
    print(header)
    print("-" * len(header))
    for i, first in enumerate(runs[0]):
        per_run = [run[i] for run in runs]
        actions = "/".join(_ABBREV.get(r["actual_action"], r["actual_action"]) for r in per_run)
        urg = "/".join(str(r["actual_urgency"]) if r["actual_urgency"] else "-" for r in per_run)
        ok = sum(r["action_match"] for r in per_run)
        exp = _ABBREV.get(first["expected_action"], first["expected_action"])
        print(f"{first['ticket_id']:<34}{exp:<6}{actions:<30}{urg:<12}{ok}/{n_runs}")

    def _metrics_block(title: str, subset: list[list[dict]]) -> None:
        print(f"\n{title} (n={len(subset[0])} tickets x {n_runs} runs)")
        summaries = [summarize(rows) for rows in subset]
        for key in summaries[0]:
            vals = [s[key] for s in summaries if s[key] is not None]
            if vals:
                print(f"  {key:<22}{_spread(vals)}")

    _metrics_block("Per-run metrics, all tickets", runs)
    _metrics_block(
        f"Per-run metrics, original {ORIGINAL_TICKET_COUNT} tickets only",
        [rows[:ORIGINAL_TICKET_COUNT] for rows in runs],
    )

    pooled = [r for rows in runs for r in rows]
    n = len(pooled)

    def _rate(label: str, pred) -> None:
        k = sum(1 for r in pooled if pred(r))
        print(f"  {label:<44}{k:>4}/{n} ({k / n:.1%})")

    print(f"\nFailure-mode rates, pooled over all ticket-runs (n={n})")
    _rate("over-escalation (exp auto, got escalated)",
          lambda r: r["expected_action"] == "auto_resolved" and r["actual_action"] == "escalated")
    _rate("under-escalation (exp escalated, got auto)",
          lambda r: r["expected_action"] == "escalated" and r["actual_action"] == "auto_resolved")
    _rate("escalated_due_to_error (transport)", lambda r: r["is_transport_error"])
    _rate("escalated_output_blocked", lambda r: r["is_output_blocked"])
    _rate("escalated_incomplete_lookup", lambda r: r["is_incomplete_lookup"])
    _rate("http_error (request failed)", lambda r: r["is_http_error"])
    _rate("category mismatch",
          lambda r: not r["is_http_error"] and r["expected_category"] != r["actual_category"])
    _rate("guardrail mismatch", lambda r: not r["guardrail_match"])

    latencies = sorted(r["latency_ms"] for r in pooled if r["latency_ms"] is not None)
    if latencies:
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        print(f"\nServer-reported latency: p50 {p50:.0f} ms, p95 {p95:.0f} ms (n={len(latencies)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", type=Path, help="also dump raw per-run rows here")
    args = parser.parse_args()

    tickets = json.loads(TICKETS_PATH.read_text())
    runs = []
    for run in range(1, args.runs + 1):
        rows = run_eval(tickets, run=run, workers=args.workers)
        print(f"\n=== Run {run}/{args.runs} ===")
        print_table(rows)
        runs.append(rows)

    print_aggregate(runs)
    if args.json:
        args.json.write_text(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
