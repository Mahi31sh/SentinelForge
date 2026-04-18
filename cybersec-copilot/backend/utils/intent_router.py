"""
Intent Router
-------------
Reads the user's message and decides which engine to activate.

Phase 1: keyword-based (simple, works well enough to start)
Phase 2: upgrade to a zero-shot classifier using transformers
"""

import re
from enum import Enum


class Intent(str, Enum):
    PHISHING    = "phishing"
    LOG_ANALYSIS = "log_analysis"
    KNOWLEDGE   = "knowledge"
    UNKNOWN     = "unknown"


# --- keyword patterns (easy to extend) ---
_PHISHING_PATTERNS = [
    r"check\s+this\s+(email|mail|link|url|message)",
    r"is\s+this\s+(phishing|spam|scam|fake|legit)",
    r"http[s]?://\S+",          # bare URL pasted in
    r"suspicious\s+(email|link|url|domain|message)",
    r"verify\s+(this|my)\s+(email|account)",
]

_LOG_PATTERNS = [
    r"(failed\s+login|authentication\s+failure|brute\s+force)",
    r"(system\s+log|access\s+log|server\s+log|error\s+log)",
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",   # IP address pasted
    r"(analyse|analyze|check)\s+(this\s+)?log",
    r"(unusual|suspicious)\s+(access|activity|traffic)",
    r"\b(SSH|RDP|FTP|HTTP)\s+\d{3}\b",         # protocol + status code
]

_KNOWLEDGE_PATTERNS = [
    r"(what\s+is|explain|how\s+does|tell\s+me\s+about)",
    r"(difference\s+between|compare)",
    r"(XSS|SQL\s+injection|CSRF|MITM|CVE|OWASP|zero.day)",
    r"(how\s+to\s+(protect|defend|prevent|detect))",
]


def detect_intent(user_message: str) -> Intent:
    """
    Returns the most likely Intent for the given message.
    Checks phishing → log → knowledge in priority order.
    """
    msg = user_message.lower()

    for pattern in _PHISHING_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return Intent.PHISHING

    for pattern in _LOG_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return Intent.LOG_ANALYSIS

    for pattern in _KNOWLEDGE_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return Intent.KNOWLEDGE

    # Default: treat as a general knowledge question
    return Intent.KNOWLEDGE
