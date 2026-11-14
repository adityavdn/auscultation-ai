"""Patient/doctor assistant on a free LLM (Vercel AI Gateway or Groq). Red flags are caught before any LLM call."""
import json
import logging
import os
import re

import httpx

log = logging.getLogger("agent")
GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
GATEWAY_MODEL = os.environ.get("GATEWAY_MODEL", "meta/llama-3.3-70b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_TURNS = 30
RED_FLAGS = re.compile(
    r"chest pain|crushing|tight(ness)? (in|across) (my )?chest|can'?t breathe|cannot breathe|"
    r"struggling to breathe|severe(ly)? breathless|blue lips|lips (are|turning) blue|faint(ed|ing)?|"
    r"passed out|unconscious|cough(ing)? (up )?blood|blood when i cough|collapse", re.I)
RED_FLAG_REPLY = ("Your symptoms could be serious. Call 999 now, or go to A&E. "
                  "If you're not sure, call NHS 111. Don't wait for an online reply.")
UNAVAILABLE = "The assistant is unavailable right now. Your results are still shown, and you can send them to the doctor."
SAFETY = (
    "You are a supportive assistant inside a RESEARCH PROTOTYPE for heart and lung sound analysis. "
    "It is decision support only and not a medical device. Rules: never state a diagnosis as fact; "
    "never give treatment, medication or dosing advice; describe AI findings as 'the model suggests' with "
    "their confidence; keep replies under 120 words in plain UK English; if symptoms sound urgent, tell them "
    "to call 999 or NHS 111.")
ROLE = {
    "patient": "You talk with a patient. Ask one short question at a time to learn symptoms, duration, "
               "smoking, past heart or lung problems and medicines, then offer to send a summary to the doctor.",
    "doctor": "You talk with a clinician. Be concise and technical, explain what each model head was trained "
              "on and its limits, and suggest what to examine or test next.",
}
SUMMARY_PROMPT = ("Summarise this conversation and the AI findings for a doctor. Reply with JSON only, keys: "
                  "symptoms, duration, history, ai_findings, urgency (one of routine, soon, urgent).")


class LLMError(RuntimeError):
    pass


def _provider():
    """Vercel AI Gateway (keyless on Vercel via OIDC) first, Groq second, else None."""
    gw = os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get("VERCEL_OIDC_TOKEN")
    if gw:
        return GATEWAY_URL, gw, GATEWAY_MODEL
    if os.environ.get("GROQ_API_KEY"):
        return GROQ_URL, os.environ["GROQ_API_KEY"], GROQ_MODEL
    return None


def _llm(messages, json_mode=False):
    url, key, model = _provider()
    body = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 400}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    try:
        r = httpx.post(url, json=body, timeout=30, headers={"Authorization": f"Bearer {key}"})
        if r.status_code >= 400:
            raise LLMError(f"{r.status_code}: {r.text[:300]}")
        text = r.json()["choices"][0]["message"]["content"]
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()) if json_mode else text
    except (httpx.HTTPError, KeyError, ValueError) as e:
        raise LLMError(str(e)) from e


def _user_text(history):
    return " ".join(m.get("content", "") for m in history if m.get("role") == "user")


def _clean(history):
    return [{"role": m["role"], "content": str(m["content"])[:2000]}
            for m in history[-MAX_TURNS:] if m.get("role") in ("user", "assistant")]


def chat(history, analysis, role):
    last = history[-1]["content"] if history else ""
    if role == "patient" and RED_FLAGS.search(last):
        return {"reply": RED_FLAG_REPLY, "red_flag": True}
    if _provider() is None:
        return {"reply": UNAVAILABLE, "red_flag": False}
    system = f"{SAFETY}\n{ROLE[role]}\nLatest AI analysis (may be empty): {json.dumps(analysis)[:3000]}"
    try:
        return {"reply": _llm([{"role": "system", "content": system}] + _clean(history)), "red_flag": False}
    except LLMError as e:
        log.warning("llm error: %s", e)
        return {"reply": UNAVAILABLE, "red_flag": False}


def summarize_case(history, analysis):
    red = bool(RED_FLAGS.search(_user_text(history)))
    base = {"symptoms": _user_text(history)[:500], "duration": "", "history": "",
            "ai_findings": (analysis or {}).get("summary", "") if isinstance(analysis, dict) else "",
            "urgency": "urgent" if red else "routine"}
    if _provider() is not None:
        try:
            got = json.loads(_llm([{"role": "system", "content": f"{SAFETY}\n{SUMMARY_PROMPT}"},
                                   {"role": "user", "content": json.dumps({"conversation": _clean(history),
                                                                           "analysis": analysis})[:6000]}],
                                  json_mode=True))
            base |= {k: str(got.get(k, base[k])) for k in base}
        except (LLMError, ValueError) as e:
            log.warning("summary fallback: %s", e)
    if base["urgency"] not in ("routine", "soon", "urgent"):
        base["urgency"] = "routine"
    if red:
        base["urgency"] = "urgent"
    return base
