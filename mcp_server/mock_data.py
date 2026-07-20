"""Hardcoded mock account data - simulates a CRM/billing lookup with no real
database or external API."""

_ACCOUNTS = {
    "acc-1001": {
        "account_id": "acc-1001",
        "plan_tier": "pro",
        "account_standing": "good",
        "open_disputes": 0,
    },
    "acc-1002": {
        "account_id": "acc-1002",
        "plan_tier": "free",
        "account_standing": "past_due",
        "open_disputes": 1,
    },
    "acc-1003": {
        "account_id": "acc-1003",
        "plan_tier": "enterprise",
        "account_standing": "good",
        "open_disputes": 0,
    },
    "acc-1004": {
        "account_id": "acc-1004",
        "plan_tier": "pro",
        "account_standing": "suspended",
        "open_disputes": 2,
    },
}


def lookup_account(account_id: str) -> dict:
    if account_id in _ACCOUNTS:
        return _ACCOUNTS[account_id]
    return {"account_id": account_id, "error": "not_found"}
