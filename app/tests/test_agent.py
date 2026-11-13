import json

import pytest

from app import agent


@pytest.fixture()
def fake_llm(monkeypatch):
    calls = []

    def fake(messages, json_mode=False):
        calls.append(messages)
        if json_mode:
            return json.dumps({"symptoms": "cough 3 days", "duration": "3 days", "history": "smoker",
                               "ai_findings": "crackles", "urgency": "soon"})
        return "Thanks. How long have you had the cough?"
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(agent, "_llm", fake)
    return calls


def test_red_flag_short_circuits(fake_llm):
    r = agent.chat([{"role": "user", "content": "I have crushing chest pain"}], None, "patient")
    assert r["red_flag"] and "999" in r["reply"]
    assert fake_llm == []


def test_normal_chat_calls_llm_with_safety_prompt(fake_llm):
    r = agent.chat([{"role": "user", "content": "I've had a cough"}], {"status": "abnormal"}, "patient")
    assert not r["red_flag"] and r["reply"]
    system = fake_llm[0][0]["content"]
    assert "never" in system.lower() and "diagnos" in system.lower()


def test_no_key_graceful(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    r = agent.chat([{"role": "user", "content": "hello"}], None, "patient")
    assert "unavailable" in r["reply"].lower() and r["red_flag"] is False


def test_llm_error_graceful(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")

    def boom(messages, json_mode=False):
        raise agent.LLMError("rate limited")
    monkeypatch.setattr(agent, "_llm", boom)
    assert "unavailable" in agent.chat([{"role": "user", "content": "hi"}], None, "doctor")["reply"].lower()


def test_summary_schema(fake_llm):
    s = agent.summarize_case([{"role": "user", "content": "cough"}], {"status": "abnormal"})
    assert set(s) == {"symptoms", "duration", "history", "ai_findings", "urgency"}
    assert s["urgency"] in {"routine", "soon", "urgent"}


def test_summary_red_flag_forces_urgent(fake_llm):
    s = agent.summarize_case([{"role": "user", "content": "I am coughing up blood"}], None)
    assert s["urgency"] == "urgent"
