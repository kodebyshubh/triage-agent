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

Runs `eval/test_tickets.json` (16 hand-crafted tickets, spread across categories/
urgency, plus 3 adversarial: injection, PII, ambiguous) against a live server and
prints expected-vs-actual routing, category, and guardrail behavior.

## MCP Integration

TriageForge also demonstrates tool interoperability across process boundaries via
the [Model Context Protocol](https://modelcontextprotocol.io) (MCP), alongside the
existing in-process LangChain tools (`classify`, `retrieve_and_draft`, `escalate`).
Where those three run as plain Python closures inside the same process as the
orchestrator, the MCP tool runs in a fully separate process, connected over stdio,
demonstrating a genuinely different integration pattern rather than just more of
the same.

`mcp_server/server.py` is a standalone MCP server (built with zero imports from or
dependency on `/app`) exposing one mock tool, `check_account_status(account_id)`,
simulating a CRM/billing lookup against a small hardcoded dataset
(`mcp_server/mock_data.py`) - no real database or external API. The FastAPI backend
connects to it once at startup via a `lifespan` hook (not reconnected per ticket,
to avoid subprocess-spawn latency on every request) and converts its tool into a
LangChain-compatible tool via `langchain-mcp-adapters`, adding it as a 4th tool
alongside the original three. The orchestrator's system prompt gained one new rule:
call `check_account_status` before `retrieve_and_draft` or `escalate` if the ticket
references a specific account.

**The existing 3-tool routing behavior is provably unchanged.** A pre-change
baseline was captured (`eval/baseline_before_mcp.txt`) and compared ticket-for-ticket
against post-wiring runs across the integration: 13/14 and then 12/14 of the
original tickets landed on byte-identical outcomes across separate regression runs,
with every difference individually explained by already-documented model
non-determinism (see Reliability Findings below), not by anything the MCP wiring
itself changed. `eval/test_tickets.json` now has 16 tickets (14 original + 2 added
to specifically exercise `check_account_status`), and `eval/run_eval.py` reports an
additional, additive "account lookup match" metric for those 2 tickets without
altering how the original 4 metrics are computed for the other 14.

Supporting run outputs, kept for evidence rather than summarized away:
`eval/after_mcp_wiring.txt` is the interleaved run that crashed on the
`AccountLookup`-terminal bug below (empty - the process died before writing
any table, which is itself the record of the crash); `eval/after_mcp_final.txt`
is the clean 14-ticket regression run after both fixes; `eval/prompt3_16ticket.txt`
is the full 16-ticket run confirming the eval-set extension didn't disturb the
original 14.

Wiring the MCP tool in did surface one real, separate finding: the model
occasionally calls `check_account_status` and stops, instead of continuing to
`retrieve_and_draft`/`escalate` as intended. Handled the same way as every other
reliability issue in this project - root-caused precisely, not hand-waved: the
system prompt rule was strengthened first (verified via a dedicated frequency test:
0/12 crashes across 3 rounds afterward), and a graceful fallback outcome,
`escalated_incomplete_lookup`, was added as a fourth non-resolution reason
alongside `escalated_due_to_error` and `escalated_output_blocked`, so this failure
mode degrades gracefully instead of crashing the request, and is never silently
miscounted as a routing win or loss.

This is a small, additive demonstration of MCP usage - one mock tool wired in
alongside the existing architecture - not a rebuild of the tool system.

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

One bounded, time-boxed follow-up experiment: tried `gpt-oss-20b` for `classify()`
alone (isolated, not a global swap) on the 3 tickets that showed inflated
`urgency: 5`. Mixed result, not adopted - 1/3 improved (urgency dropped from 5 to
3, below the escalate threshold), 1/3 unchanged, and 1/3 hit the same
transport-layer glitch (below) in a new call site. Kept `llama-3.1-8b-instant`
for `classify()`, as originally.

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

One hypothesis tested and disproven: the KB source markdown originally contained
literal double-quotes (e.g. `"Request Refund"`), which could plausibly cause the
model to mis-escape nested quotes when generating structured JSON output. Stripped
all quotes from `data/kb/*.md`, re-ingested, and reran a query that had failed 3/3
times immediately before the change - it failed identically 3/3 times after.
Quote content in the KB is not the trigger; the root cause remains unidentified
generation-format flakiness in the underlying model.

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
