from app.llm import get_chat_model, invoke_structured_with_retry
from app.schemas import ClassificationResult, TicketInput

_SYSTEM_PROMPT = (
    "You are a support ticket classifier. Given a ticket message, determine its "
    "category, urgency (1-5, where 5 is most urgent), and sentiment."
)


def classify(ticket: TicketInput) -> ClassificationResult:
    llm = get_chat_model(reasoning_effort="low")
    structured_llm = llm.with_structured_output(
        ClassificationResult, method="json_schema", strict=True
    )
    return invoke_structured_with_retry(
        structured_llm,
        [
            ("system", _SYSTEM_PROMPT),
            ("human", ticket.user_message),
        ],
    )
