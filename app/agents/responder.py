from app.schemas import ResolutionDraft, RetrievalResult, TicketInput
from app.llm import get_chat_model, invoke_structured_with_retry

_SYSTEM_PROMPT = (
    "You are a support response drafter. Using ONLY the provided knowledge base "
    "excerpts, draft a helpful reply to the ticket. Cite which source file(s) you "
    "used in the citations field. If the excerpts don't answer the question, say "
    "so in draft_text and give a low confidence score."
)


def draft_response(ticket: TicketInput, retrieval: RetrievalResult) -> ResolutionDraft:
    llm = get_chat_model()
    structured_llm = llm.with_structured_output(ResolutionDraft)

    context = "\n\n".join(
        f"[Source: {src}]\n{chunk}"
        for src, chunk in zip(retrieval.sources, retrieval.chunks)
    )
    human_prompt = (
        f"Ticket message: {ticket.user_message}\n\n"
        f"Knowledge base excerpts:\n{context}"
    )

    return invoke_structured_with_retry(
        structured_llm,
        [
            ("system", _SYSTEM_PROMPT),
            ("human", human_prompt),
        ],
    )
