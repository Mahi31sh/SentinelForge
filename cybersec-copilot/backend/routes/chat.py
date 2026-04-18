from __future__ import annotations

import re
import uuid
from typing import Any

from fastapi import APIRouter
from fastapi import Query
from pydantic import BaseModel

from backend.utils.intent_router import Intent, detect_intent
from backend.utils.chat_history_store import append_chat_exchange, fetch_chat_history
from backend.engines import explainer_engine
from backend.engines import log_engine, phishing_engine
from backend.engines.threat_intel import check_ip_abuse, check_url_virustotal

router = APIRouter()

try:
    from backend.engines import knowledge_engine

    _HAS_RAG = True
except Exception:
    _HAS_RAG = False

try:
    from backend.engines import ml_phishing_engine

    _HAS_ML_PHISH = True
except Exception:
    _HAS_ML_PHISH = False

try:
    from backend.engines import ml_log_engine

    _HAS_ML_LOG = True
except Exception:
    _HAS_ML_LOG = False


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] | None = None


class IOCItem(BaseModel):
    indicator: str
    indicator_type: str
    provider: str
    status: str
    score: int | None
    summary: str


class ChatResponse(BaseModel):
    intent: str
    verdict: str | None = None
    confidence: float | None = None
    severity: str | None = None
    findings_count: int | None = None
    mitre_tactics: list[str] | None = None
    explanation: str
    raw_features: list[str] | None = None
    ioc_enrichment: list[IOCItem] | None = None
    anomaly_score: float | None = None
    anomaly_level: str | None = None
    detection_method: str | None = None
    audit_id: str | None = None


class ChatHistoryItem(BaseModel):
    id: int
    created_at: str
    audit_id: str | None = None
    intent: str
    message: str
    response: ChatResponse


def extract_ip(text: str) -> str | None:
    match = re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)
    return match.group(0) if match else None


def extract_url(text: str) -> str | None:
    match = re.search(r"https?://\S+", text)
    return match.group(0) if match else None


class _Phase1Wrapper:
    def __init__(self, result: Any):
        self.verdict = result.verdict
        self.confidence = result.confidence
        self.features = result.features
        self.urls_found = result.urls_found


def _wrap_phase1_phishing(result: Any) -> _Phase1Wrapper:
    return _Phase1Wrapper(result)


def _highest_severity(findings: list[Any]) -> str | None:
    order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    if not findings:
        return None
    return max(findings, key=lambda finding: order.get(finding.severity, 0)).severity


def _ioc_status_from_score(score: int) -> str:
    if score >= 80:
        return "malicious"
    if score >= 40:
        return "suspicious"
    return "clean"


def _run_ioc(msg: str, intent: Intent) -> list[IOCItem] | None:
    if intent == Intent.KNOWLEDGE:
        return None

    url = extract_url(msg)
    ip = extract_ip(msg)
    items: list[IOCItem] = []

    if url:
        try:
            vt_result = check_url_virustotal(url)
        except Exception:
            vt_result = None

        if vt_result:
            malicious = int(vt_result.get("malicious", 0) or 0)
            suspicious = int(vt_result.get("suspicious", 0) or 0)
            harmless = int(vt_result.get("harmless", 0) or 0)

            if malicious > 0:
                status = "malicious"
            elif suspicious > 0:
                status = "suspicious"
            else:
                status = "clean"

            score = min(100, malicious * 10 + suspicious * 5)
            items.append(
                IOCItem(
                    indicator=url,
                    indicator_type="url",
                    provider="VirusTotal",
                    status=status,
                    score=score,
                    summary=(
                        f"Malicious: {malicious}, Suspicious: {suspicious}, Harmless: {harmless}"
                    ),
                )
            )

    if ip:
        try:
            abuse_result = check_ip_abuse(ip)
        except Exception:
            abuse_result = None

        if abuse_result:
            score = (
                abuse_result.get("data", {}).get("abuseConfidenceScore")
                if "data" in abuse_result
                else abuse_result.get("abuseConfidenceScore", 0)
            ) or 0

            items.append(
                IOCItem(
                    indicator=ip,
                    indicator_type="ip",
                    provider="AbuseIPDB",
                    status=_ioc_status_from_score(int(score)),
                    score=int(score),
                    summary=f"Abuse confidence score: {int(score)}",
                )
            )

    return items or None


def _merge_ioc_severity(base_severity: str | None, ioc_items: list[IOCItem] | None) -> str | None:
    order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    merged = base_severity

    if ioc_items:
        if any(item.status == "malicious" for item in ioc_items):
            merged = "high" if not merged or order.get(merged, 0) < order["high"] else merged
        elif any(item.status == "suspicious" for item in ioc_items):
            merged = "medium" if not merged or order.get(merged, 0) < order["medium"] else merged

    return merged


def _merge_ioc_phishing_result(result: Any, ioc_items: list[IOCItem] | None) -> Any:
    """Upgrade phishing verdict/confidence when IOC data is more severe than model output."""
    if not ioc_items:
        return result

    has_malicious = any(item.status == "malicious" for item in ioc_items)
    has_suspicious = any(item.status == "suspicious" for item in ioc_items)

    # Never downgrade model/rules verdicts based on IOC; only escalate certainty.
    if has_malicious and result.verdict != "phishing":
        result.verdict = "phishing"
        result.confidence = max(float(result.confidence or 0.0), 0.9)
    elif has_suspicious and result.verdict == "clean":
        result.verdict = "suspicious"
        result.confidence = max(float(result.confidence or 0.0), 0.6)

    if hasattr(result, "features") and isinstance(result.features, list):
        for item in ioc_items:
            if item.status == "malicious":
                result.features.append(
                    f"IOC enrichment ({item.provider}) flagged indicator as MALICIOUS: {item.summary}"
                )
            elif item.status == "suspicious":
                result.features.append(
                    f"IOC enrichment ({item.provider}) flagged indicator as SUSPICIOUS: {item.summary}"
                )

    return result


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    msg = req.message.strip()
    intent = detect_intent(msg)
    audit_id = str(uuid.uuid4())[:8].upper()

    ioc_items = _run_ioc(msg, intent)

    if intent == Intent.PHISHING:
        if _HAS_ML_PHISH and ml_phishing_engine.model_available():
            result = ml_phishing_engine.analyse(msg)
            method = "ml"
        else:
            raw = phishing_engine.analyse(msg)
            result = _wrap_phase1_phishing(raw)
            method = "rules"

        result = _merge_ioc_phishing_result(result, ioc_items)

        explanation = explainer_engine.explain_phishing(msg, result, history=req.history)
        response = ChatResponse(
            intent=intent,
            verdict=result.verdict,
            confidence=result.confidence,
            explanation=explanation,
            raw_features=result.features,
            ioc_enrichment=ioc_items,
            detection_method=method,
            audit_id=audit_id,
        )
        append_chat_exchange(msg, response)
        return response
    
   

    if intent == Intent.LOG_ANALYSIS:
        rule_result = log_engine.analyse(msg)
        top_severity = _highest_severity(rule_result.findings)
        top_severity = _merge_ioc_severity(top_severity, ioc_items)

        tactics = list({finding.mitre_tactic for finding in rule_result.findings if finding.mitre_tactic})

        anomaly_score = None
        anomaly_level = None
        method = "rules"

        if _HAS_ML_LOG:
            try:
                anomaly = ml_log_engine.analyse_anomaly(msg)
                anomaly_score = anomaly.anomaly_score
                anomaly_level = anomaly.anomaly_level
                method = anomaly.method
            except Exception:
                pass

        explanation = explainer_engine.explain_log_analysis(msg, rule_result, history=req.history)

        response = ChatResponse(
            intent=intent,
            severity=top_severity,
            findings_count=len(rule_result.findings),
            mitre_tactics=tactics,
            explanation=explanation,
            ioc_enrichment=ioc_items,
            anomaly_score=anomaly_score,
            anomaly_level=anomaly_level,
            detection_method=method,
            audit_id=audit_id,
        )
        append_chat_exchange(msg, response)
        return response

    context = knowledge_engine.query(msg) if _HAS_RAG else []
    explanation = explainer_engine.explain_knowledge_query(msg, context, history=req.history)
    response = ChatResponse(
        intent=intent,
        explanation=explanation,
        ioc_enrichment=ioc_items,
        detection_method="rag" if (_HAS_RAG and context) else "llm",
        audit_id=audit_id,
    )
    append_chat_exchange(msg, response)
    return response


@router.get("/chat/history", response_model=list[ChatHistoryItem])
def chat_history(limit: int = Query(default=50, ge=1, le=200)) -> list[ChatHistoryItem]:
    items = fetch_chat_history(limit=limit)
    return [ChatHistoryItem(**item) for item in items]