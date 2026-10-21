from app import llm


def _set_keys(monkeypatch, primary, extras):
    monkeypatch.setenv("GROQ_API_KEY", primary)
    for i in range(1, 11):
        monkeypatch.setenv(f"GROQ_API_KEY_{i}", extras.get(i, ""))


def test_empty_and_whitespace_slots_are_ignored(monkeypatch):
    _set_keys(monkeypatch, "k0", {1: "  ", 2: "k2 \r", 5: "k5"})
    assert llm.groq_keys() == ["k0", "k2", "k5"]


def test_duplicate_keys_dropped(monkeypatch):
    _set_keys(monkeypatch, "k0", {2: "k0", 3: "k3"})
    assert llm.groq_keys() == ["k0", "k3"]


def test_round_robin_covers_every_key(monkeypatch):
    _set_keys(monkeypatch, "k0", {2: "k2", 3: "k3"})
    picked = [llm.next_groq_key() for _ in range(6)]
    assert sorted(set(picked)) == ["k0", "k2", "k3"]
    assert picked[:3] == picked[3:]


def test_no_keys_returns_none(monkeypatch):
    _set_keys(monkeypatch, "", {})
    assert llm.next_groq_key() is None
