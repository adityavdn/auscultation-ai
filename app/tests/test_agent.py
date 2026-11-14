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


def test_no_key_falls_back_to_scripted(monkeypatch):
    for k in ("VERCEL_OIDC_TOKEN", "AI_GATEWAY_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    r = agent.chat([{"role": "user", "content": "hello"}], None, "patient")
    assert "how long" in r["reply"].lower() and r["red_flag"] is False


def test_llm_error_graceful(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")

    def boom(messages, json_mode=False):
        raise agent.LLMError("rate limited")
    monkeypatch.setattr(agent, "_llm", boom)
    assert "unavailable" not in agent.chat([{"role": "user", "content": "hi"}], None, "patient")["reply"].lower()


def test_summary_schema(fake_llm):
    s = agent.summarize_case([{"role": "user", "content": "cough"}], {"status": "abnormal"})
    assert set(s) == {"symptoms", "duration", "history", "ai_findings", "urgency"}
    assert s["urgency"] in {"routine", "soon", "urgent"}


def test_summary_red_flag_forces_urgent(fake_llm):
    s = agent.summarize_case([{"role": "user", "content": "I am coughing up blood"}], None)
    assert s["urgency"] == "urgent"


def test_provider_prefers_vercel_gateway(monkeypatch):
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc")
    monkeypatch.setenv("GROQ_API_KEY", "groq")
    url, key, _ = agent._provider()
    assert "ai-gateway.vercel.sh" in url and key == "oidc"


def test_provider_falls_back_to_groq(monkeypatch):
    for k in ("VERCEL_OIDC_TOKEN", "AI_GATEWAY_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "groq")
    url, key, _ = agent._provider()
    assert "groq" in url and key == "groq"


def test_no_provider_is_unavailable(monkeypatch):
    for k in ("VERCEL_OIDC_TOKEN", "AI_GATEWAY_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert agent._provider() is None



ANSWERS = ["cough and wheeze", "two weeks", "I smoked for 10 years", "asthma as a child", "salbutamol inhaler"]


def _hist(n):
    h = []
    for a in ANSWERS[:n]:
        h += [{"role": "user", "content": a}, {"role": "assistant", "content": "ok"}]
    return h[:-1]


@pytest.fixture()
def no_llm(monkeypatch):
    for k in ("VERCEL_OIDC_TOKEN", "AI_GATEWAY_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("n,expected", [(1, "how long"), (2, "smoke"), (3, "heart or lung"),
                                        (4, "medicines"), (5, "send summary")])
def test_scripted_asks_questions_in_order(no_llm, n, expected):
    assert expected in agent.chat(_hist(n), None, "patient")["reply"].lower()


def test_scripted_final_reply_explains_findings_without_diagnosing(no_llm):
    a = {"fusion": {"status": "abnormal"}, "summary": "lung sound: wheeze (71%)"}
    reply = agent.chat(_hist(5), a, "patient")["reply"]
    assert "wheeze (71%)" in reply and "not a diagnosis" in reply.lower()


def test_scripted_doctor_reply_lists_findings(no_llm):
    a = {"fusion": {"status": "abnormal"}, "summary": "heart murmur: present (82%)"}
    assert "heart murmur: present (82%)" in agent.chat([{"role": "user", "content": "findings?"}], a, "doctor")["reply"]


def test_scripted_summary_maps_answers(no_llm):
    s = agent.summarize_case(_hist(5), {"fusion": {"status": "abnormal"}, "summary": "lung sound: wheeze"})
    assert s["symptoms"] == ANSWERS[0] and s["duration"] == ANSWERS[1]
    assert all(a in s["history"] for a in ANSWERS[2:])
    assert s["ai_findings"] == "lung sound: wheeze" and s["urgency"] == "soon"


def test_scripted_summary_routine_when_normal(no_llm):
    assert agent.summarize_case(_hist(2), {"fusion": {"status": "normal"}, "summary": ""})["urgency"] == "routine"
