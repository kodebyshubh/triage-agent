import pytest
from pydantic import ValidationError

from app.schemas import (
    ClassificationResult,
    EscalationSummary,
    ResolutionDraft,
    RetrievalResult,
    TicketInput,
    TriageResponse,
)


def test_ticket_input_requires_all_fields():
    with pytest.raises(ValidationError):
        TicketInput(ticket_id="t1", session_id="s1")


@pytest.mark.parametrize("urgency", [1, 3, 5])
def test_classification_accepts_urgency_in_range(urgency):
    c = ClassificationResult(category="billing", urgency=urgency, sentiment="neutral")
    assert c.urgency == urgency


@pytest.mark.parametrize("urgency", [0, 6, -1])
def test_classification_rejects_urgency_out_of_range(urgency):
    with pytest.raises(ValidationError):
        ClassificationResult(category="billing", urgency=urgency, sentiment="neutral")


def test_classification_rejects_unknown_category_and_sentiment():
    with pytest.raises(ValidationError):
        ClassificationResult(category="shipping", urgency=2, sentiment="neutral")
    with pytest.raises(ValidationError):
        ClassificationResult(category="billing", urgency=2, sentiment="happy")


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_bounds_enforced(confidence):
    with pytest.raises(ValidationError):
        RetrievalResult(chunks=[], sources=[], confidence=confidence)
    with pytest.raises(ValidationError):
        ResolutionDraft(draft_text="x", confidence=confidence, citations=[])


def test_escalation_priority_literal():
    with pytest.raises(ValidationError):
        EscalationSummary(summary="s", suggested_priority="urgent", flagged_reason="r")


@pytest.mark.parametrize(
    "action",
    [
        "auto_resolved",
        "escalated",
        "escalated_due_to_error",
        "escalated_output_blocked",
        "escalated_incomplete_lookup",
    ],
)
def test_triage_response_accepts_each_distinct_final_action(action):
    summary = EscalationSummary(summary="s", suggested_priority="low", flagged_reason="r")
    assert TriageResponse(final_action=action, response=summary, trace=[]).final_action == action


def test_triage_response_rejects_merged_or_unknown_action():
    summary = EscalationSummary(summary="s", suggested_priority="low", flagged_reason="r")
    with pytest.raises(ValidationError):
        TriageResponse(final_action="failed", response=summary, trace=[])
