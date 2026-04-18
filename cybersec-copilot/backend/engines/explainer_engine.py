"""LLM explainer using Groq with deterministic local fallback."""

import os
import requests

from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
except Exception:
    Groq = None


SYSTEM_PROMPT = """You are an expert cybersecurity analyst assistant.
Your job is to explain security findings clearly to non-technical users.
Be conversational, practical, and concise.
Use natural chatbot language instead of rigid templates.
When relevant, include:
- what happened
- why it matters
- what to do next
If the user asks follow-up questions, keep context from prior turns."""


class LLMUnavailableError(RuntimeError):
    """Raised when no LLM provider can generate a response."""


def _build_client():
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()

    if groq_key and Groq is not None:
        return Groq(api_key=groq_key)
    return None


def explain_phishing(text: str, result, history: list[dict[str, str]] | None = None) -> str:
    """Generate an LLM explanation for a phishing detection result."""
    features_text = "\n".join(f"- {f}" for f in result.features) or "- No specific signals found"
    urls_text = "\n".join(result.urls_found) if result.urls_found else "None"

    prompt = f"""A user submitted this text for phishing analysis:

{text[:1500]}

Our automated system found these suspicious signals:
{features_text}

URLs detected: {urls_text}
Verdict: {result.verdict.upper()} (confidence: {int(result.confidence * 100)}%)

Explain this result to the user. Include:
- why each signal is suspicious
- what the attacker's goal likely is
- exactly what the user should do next
"""

    try:
        return _call_llm(prompt, history=history)
    except LLMUnavailableError:
        return _llm_unavailable_message()


def explain_log_analysis(log_text: str, result, history: list[dict[str, str]] | None = None) -> str:
    """Generate an LLM explanation for log analysis findings."""
    if not result.findings:
        prompt = f"""A user submitted system logs for security analysis.

{result.summary}

No direct rule-based findings were detected.

Respond as a helpful SOC assistant. Explain:
- what this means and what could still be missed
- what practical checks the user should run next
- when they should escalate
"""
        try:
            return _call_llm(prompt, history=history)
        except LLMUnavailableError:
            return _llm_unavailable_message()

    findings_text = ""
    for f in result.findings:
        findings_text += (
            f"\n**{f.attack_type}** (Severity: {f.severity.upper()})\n"
            f"MITRE ATT&CK: {f.mitre_tactic} — {f.mitre_name}\n"
            f"Evidence:\n" + "\n".join(f"  - {e}" for e in f.evidence) + "\n"
        )

    prompt = f"""A user submitted system logs for security analysis.

{result.summary}

Findings:
{findings_text}

Explain what is happening in plain English. For each finding, describe:
- what the attacker is trying to do
- how serious this is
- the specific steps the user should take RIGHT NOW to respond
"""

    try:
        return _call_llm(prompt, history=history)
    except LLMUnavailableError:
        return _llm_unavailable_message()


def explain_knowledge_query(
    question: str,
    context_chunks: list[str],
    history: list[dict[str, str]] | None = None,
) -> str:
    """Answer a cybersecurity question using retrieved context (RAG)."""
    context = "\n\n".join(context_chunks) if context_chunks else "No specific context retrieved."

    prompt = f"""Answer this cybersecurity question using the provided context.

Question: {question}

Context from knowledge base:
{context}

If context is insufficient, supplement with general cybersecurity knowledge.
Give a clear explanation with an example if helpful.
"""

    try:
        return _call_llm(prompt, history=history)
    except LLMUnavailableError:
        return _llm_unavailable_message()


def _normalize_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    if not history:
        return []

    normalized: list[dict[str, str]] = []
    for turn in history[-12:]:
        role = (turn.get("role") or "").strip().lower()
        content = (turn.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content[:3000]})

    return normalized


def _call_llm(user_prompt: str, history: list[dict[str, str]] | None = None) -> str | None:
    normalized_history = _normalize_history(history)

    groq_text = _call_groq(user_prompt, normalized_history)
    if groq_text:
        return groq_text

    ollama_text = _call_ollama(user_prompt, normalized_history)
    if ollama_text:
        return ollama_text

    raise LLMUnavailableError("No active LLM provider returned a response")


def _call_groq(user_prompt: str, history: list[dict[str, str]] | None = None) -> str | None:
    client = _build_client()
    if client is None:
        return None

    models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ]
    for model in models:
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": user_prompt})

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
        except Exception:
            continue

    return None


def _call_ollama(user_prompt: str, history: list[dict[str, str]] | None = None) -> str | None:
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend({"role": turn["role"], "content": turn["content"]} for turn in (history or []))
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": ollama_model,
        "messages": messages,
        "stream": False,
    }
    try:
        response = requests.post(
            f"{ollama_base_url}/api/chat",
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        text = message.get("content", "") if isinstance(message, dict) else ""
        return text.strip() if text else None
    except Exception:
        return None


def _llm_unavailable_message() -> str:
    return (
        "LLM response is unavailable right now. Configure a valid `GROQ_API_KEY` or run Ollama and set "
        "`OLLAMA_MODEL`, then restart the server to get live chatbot responses."
    )