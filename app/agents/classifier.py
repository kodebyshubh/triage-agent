from app.llm import ROLE_MODELS, get_chat_model, invoke_structured_with_retry
from app.schemas import ClassificationResult, TicketInput

# Few-shot calibration for urgency. The eval found the classifier scoring any
# frustrated-sounding ticket as urgency 5, which then (correctly, per its rules)
# made the orchestrator escalate minor issues. These examples anchor urgency to
# the impact of the problem, not the customer's tone. They were written for this
# prompt and do not reuse eval/test_tickets.json text.
_SYSTEM_PROMPT = """You are a support ticket classifier. Given a ticket message, determine its category, urgency (1-5, where 5 is most urgent), and sentiment.

Rate urgency by the impact and time pressure of the underlying problem, NOT by how upset the customer sounds. Frustration, capital letters, exclamation marks, or words like "annoying" or "ridiculous" never by themselves justify urgency 4 or 5. Sentiment captures tone; urgency captures impact.

Urgency scale:
1 = general question or request for information, nothing is broken
2 = minor problem or how-to with a documented fix; the customer can wait
3 = something is broken or wrong for this one customer, money may be involved but is recoverable through a normal process
4 = serious impact: suspected fraud or unauthorized activity, locked out with a hard deadline today, data loss, or several people blocked
5 = critical: active security breach, outage blocking a business or many users, legal or chargeback threat with a deadline, irreversible loss

Examples:
Message: "This is ridiculous, I can't find where to change my profile picture!!"
-> category=account, urgency=2, sentiment=frustrated

Message: "Why does the export button do nothing? So irritating, I've clicked it ten times."
-> category=technical, urgency=2, sentiment=frustrated

Message: "I was billed again this month even though I cancelled in March. Really not happy about this."
-> category=billing, urgency=3, sentiment=frustrated

Message: "Can our company PO number be printed on future receipts?"
-> category=billing, urgency=1, sentiment=neutral

Message: "The app logs me out every couple of hours."
-> category=technical, urgency=2, sentiment=neutral

Message: "I just got an alert that my password was changed from another country and I did not do it."
-> category=account, urgency=5, sentiment=urgent

Message: "Checkout has been failing for every customer in our store since your update an hour ago."
-> category=technical, urgency=5, sentiment=urgent

Message: "Do you have a public API that partners can integrate with?"
-> category=other, urgency=1, sentiment=neutral
"""


def classify(ticket: TicketInput) -> ClassificationResult:
    llm = get_chat_model(model=ROLE_MODELS["classifier"], reasoning_effort="low")
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
