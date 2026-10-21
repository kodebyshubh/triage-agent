from app import llm


def test_defaults_to_current_model_for_every_role(monkeypatch):
    for role in llm.ROLES:
        monkeypatch.delenv(f"MODEL_{role.upper()}", raising=False)
    assert llm.role_models_from_env() == {role: llm.GROQ_MODEL for role in llm.ROLES}


def test_env_overrides_single_role(monkeypatch):
    for role in llm.ROLES:
        monkeypatch.delenv(f"MODEL_{role.upper()}", raising=False)
    monkeypatch.setenv("MODEL_ORCHESTRATOR", "openai/gpt-oss-120b")
    monkeypatch.setenv("MODEL_CLASSIFIER", "   ")  # blank counts as unset
    models = llm.role_models_from_env()
    assert models["orchestrator"] == "openai/gpt-oss-120b"
    assert models["classifier"] == llm.GROQ_MODEL


def test_reasoning_effort_kept_for_gpt_oss_dropped_for_others(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    gpt = llm.get_chat_model(model="openai/gpt-oss-20b", reasoning_effort="low")
    qwen = llm.get_chat_model(model="qwen/qwen3.8-27b", reasoning_effort="low")
    assert gpt.reasoning_effort == "low"
    assert qwen.reasoning_effort is None
