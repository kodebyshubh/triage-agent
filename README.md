# TriageForge

Agentic support-ticket triage system: ingests a ticket, runs it through an input
guardrail, an Orchestrator agent that genuinely delegates to Classifier /
Retriever+Responder / Escalator tools via LangChain tool-calling, then an output
guardrail, and returns a structured `TriageResponse` with a full agent trace.

Built as a portfolio project demonstrating real multi-agent delegation, a guardrail
layer, and Pydantic-enforced structured output at every agent boundary - explicitly
scoped as demo-grade, not production-grade.

## Setup

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
# add GROQ_API_KEY (and optionally GEMINI_API_KEY) to .env
venv\Scripts\python -m app.ingest        # embed data/kb/*.md into local ChromaDB
venv\Scripts\python -m uvicorn app.main:app --reload
```

## Evaluation

```
venv\Scripts\python -m eval.run_eval
```

Runs `eval/test_tickets.json` (14 hand-crafted tickets, spread across categories/
urgency, plus 3 adversarial: injection, PII, ambiguous) against a live server and
prints expected-vs-actual routing, category, and guardrail behavior.

## Reliability Findings

Running the eval set against real Groq models surfaced three distinct, unrelated
failure modes. Each was root-caused with reproducible evidence before any fix was
applied, and each was fixed at the narrowest scope that addressed it - none were
papered over or folded into another outcome.

### 1. Systematic over-escalation (routing bias)

**Baseline: 0/8 correct** on tickets that should auto-resolve via
`retrieve_and_draft` - all 9 were escalated instead, while 5/5 tickets that
should escalate did so correctly. Not random noise: a clean 100%/0% split.

Root cause, found via direct trace inspection: the **classifier**, not the
orchestrator, is the source of the bias. `classify()` reliably assigns
`urgency: 5` to any ticket with frustrated/negative tone - "can you help me sort
it out," "crashes every time," "keep getting a connection timeout error" - even
for objectively minor issues (a duplicate charge, an occasional crash, a
timeout). The orchestrator's routing rule (`escalate` only if `urgency >= 4` or
retrieval `confidence < 0.4`) is then followed correctly given that input. This
is a classifier miscalibration, not a broken orchestration loop - an important
distinction, since it means the delegation architecture itself is sound.

Fix attempts, in order:
- **Prompt tightening alone** (replacing a qualitative escalation rule with a
  mechanical, quantitative one - `urgency >= 4 OR confidence < 0.4`, "never
  escalate as a default/safe choice"): **0/8, no improvement.** Confirmed this
  was a model-capability limit, not a prompt-clarity problem.
- **Scoped model swap**: the orchestrator's tool-selection loop now runs on
  `openai/gpt-oss-20b` (proven reliable at multi-step tool-calling in isolated
  testing) while `classify`/`draft_response`/`escalate` keep using
  `llama-3.1-8b-instant` (proven reliable for structured output). This is the
  fix that moved the number: **3/6 correct (50%)** on tickets that reached a
  genuine routing decision (2/8 separately hit transport errors - see below -
  excluded from this count either way).

Not pursued further: per-call-site model creep or continued prompt engineering
on a small/fast model has diminishing returns. 50%, quantified and root-caused,
is a stronger result than either an unexplained 100% or unbounded time spent
chasing it, given this is explicitly a demo-grade project, not production-grade.

### 2. Transport-layer glitch (malformed tool-call generation)

Groq's models occasionally emit malformed function-call syntax
(`400 tool_use_failed`, e.g. `<function=escalate>{...}` instead of a real tool
call) - a generation-format bug, not a content decision. Confirmed via direct
API error inspection; happens unpredictably in both the orchestrator's own
reasoning and any structured-output call (`classify`/`draft_response`/`escalate`).

Handling: retried up to 2 extra times, scoped **only** to this specific error
(`groq.BadRequestError` containing `tool_use_failed`) - never to a syntactically
valid but disagreeable routing decision, which would quietly substitute engineered
retries for genuine model judgment. When retries are exhausted, the ticket
returns a distinct `final_action: "escalated_due_to_error"` rather than crashing
with a 500 or being counted as a routing win/loss. Still occurs (2/8 in the final
clean eval run) but is now always visible and never silently contaminates the
routing-accuracy number.

### 3. Output-guardrail false positive

The output moderation LLM check rejected a completely benign, correct
password-reset reply, apparently triggered by a line referencing the KB's
documented verification process ("...contact support with a government ID for
manual verification"). Confirmed reproducible in isolation before any fix.

Fix: added a clarifying example to the guardrail's prompt (a response
referencing a normal, documented support process is not itself a leak). Also
changed the failure behavior: a rejected output now returns a distinct
`final_action: "escalated_output_blocked"` instead of a hard 500 - a real
support system shouldn't crash when it can't safely show a drafted reply, it
should hand off to a human. Retested standalone before rerunning anything
broader: guardrail now passes the exact response that previously failed.
After the fix: 0/8 output-blocks in the clean eval rerun, and the ticket that
originally crashed on this now resolves correctly end-to-end.

### 4. Injection-classifier false positive

The LLM injection check flagged a legitimate PII-containing refund request
(`"my email is john@example.com... please refund me"`) as also being a
prompt-injection attempt, purely because it contained personal information.

Fix: added a clarifying negative example to the injection-check prompt (PII
alone is not injection). Retested: the previously-flagged ticket is no longer
rejected by the guardrail. True-positive detection is unaffected - a genuine
injection attempt in the eval set is still caught and rejected correctly.

### Summary

| Failure mode | Before | After |
|---|---|---|
| Routing bias (over-escalation) | 0/8 correct | 3/6 correct (50%) on genuine decisions |
| Transport glitch (malformed generation) | Crashed the request (500) | Isolated as `escalated_due_to_error`, never silently miscounted |
| Output-guardrail false positive | Crashed the request (500) | 0/8 blocks; affected ticket now resolves correctly |
| Injection-classifier false positive | Rejected a legitimate request | No longer rejected; true positives still caught |

All three non-resolution outcomes (`escalated`, `escalated_due_to_error`,
`escalated_output_blocked`) are kept distinct in the schema and the eval script
specifically so that conflating any pair of them can't hide which failure mode is
actually responsible for a given ticket's outcome.

### Known verification gap

`escalated_output_blocked` is confirmed working at the data/backend layer - the
guardrail fix was retested standalone against the exact string that originally
triggered it, and `main.py`'s handling of a rejected output was verified by
calling it directly. One attempt was made to trigger it end-to-end through the
live UI (a ticket asking the model to reveal its system prompt), but that
message was - correctly - caught by the *input* guardrail's injection heuristic
first, so it never reached the output-guardrail path. The rendered UI state for
`escalated_output_blocked` (the red badge/card branch in `ResultCard`) has not
been visually confirmed in a browser, since manufacturing a genuine trigger
without a live browser session proved unreliable. Session-memory context
carrying across a two-turn conversation, by contrast, **was** confirmed live:
a follow-up message containing none of the original ticket's keywords still
produced a response referencing the original topic, which can only happen via
the injected history context.
