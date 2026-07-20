"""POSTs eval/test_tickets.json to a live TriageForge server and prints a summary
table: expected vs actual routing, and whether guardrails triggered as expected.

Usage: start the server (uvicorn app.main:app), then run:
    python -m eval.run_eval
"""

import json
from pathlib import Path

import requests

TICKETS_PATH = Path(__file__).parent / "test_tickets.json"
SERVER_URL = "http://127.0.0.1:8000"


def _actual_category(trace: list[dict]) -> str | None:
    for step in trace:
        if step["agent_name"] == "Classifier":
            return step["output"].get("category")
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


def run_eval() -> list[dict]:
    tickets = json.loads(TICKETS_PATH.read_text())
    rows = []

    for t in tickets:
        payload = {
            "ticket_id": t["ticket_id"],
            "user_message": t["user_message"],
            "session_id": t["session_id"],
        }
        resp = requests.post(f"{SERVER_URL}/ticket", json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()

        expected_action = t["expected_action"]
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
        guardrail_expected = bool(t["expected_guardrail_flags"])
        guardrail_actual = _guardrail_triggered(result)

        # Additional metric (Prompt 3): only applies to tickets that declare
        # expect_account_lookup - the original 14 tickets don't have this key,
        # so lookup_match stays None for them (not applicable), never counted
        # as a pass or a fail, and never touches the 4 existing metrics above.
        expect_lookup = t.get("expect_account_lookup")
        actual_lookup = _account_lookup_called(result["trace"])
        lookup_match = None if expect_lookup is None else (expect_lookup == actual_lookup)

        rows.append(
            {
                "ticket_id": t["ticket_id"],
                "expected_action": expected_action,
                "actual_action": actual_action,
                "is_transport_error": is_transport_error,
                "is_output_blocked": is_output_blocked,
                "is_incomplete_lookup": is_incomplete_lookup,
                "action_match": (not is_non_routing_outcome) and expected_action == actual_action,
                "expected_category": t["expected_category"],
                "actual_category": _actual_category(result["trace"]),
                "guardrail_expected": guardrail_expected,
                "guardrail_actual": guardrail_actual,
                "guardrail_match": guardrail_expected == guardrail_actual,
                "expect_lookup": expect_lookup,
                "actual_lookup": actual_lookup,
                "lookup_match": lookup_match,
            }
        )

    return rows


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
    n_lookup_applicable = 0
    n_lookup_match = 0
    for r in rows:
        if r["is_transport_error"]:
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
    if n_lookup_applicable:
        print(
            f"Account lookup match: {n_lookup_match}/{n_lookup_applicable} "
            f"({n_lookup_match / n_lookup_applicable:.0%}) "
            "(only tickets declaring expect_account_lookup; additive metric, "
            "doesn't affect routing/guardrail match above)"
        )


if __name__ == "__main__":
    print_table(run_eval())
