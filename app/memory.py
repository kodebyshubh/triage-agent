"""Session memory: in-memory dict keyed by session_id.

Plain dict of Pydantic models chosen over ConversationBufferMemory - the
orchestrator's tools take a single TicketInput, not a chat-message history, so a
list of prior (TicketInput, TriageResponse) pairs is a more direct fit than
LangChain's memory classes here.

Not persisted across restarts - documented non-goal (PRD section 3/8).
"""

from app.schemas import TicketInput, TriageResponse

_store: dict[str, list[tuple[TicketInput, TriageResponse]]] = {}


def get_history(session_id: str) -> list[tuple[TicketInput, TriageResponse]]:
    return _store.get(session_id, [])


def append_turn(session_id: str, ticket: TicketInput, response: TriageResponse) -> None:
    _store.setdefault(session_id, []).append((ticket, response))
