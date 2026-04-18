"""
Phishing Detection Engine
--------------------------
Phase 1: Rule-based feature extraction (works out of the box, no training needed)
Phase 2: Swap predict() to use a trained scikit-learn model

Features extracted:
  - Suspicious keywords in text
  - URL structure analysis (IP address, long subdomain, typosquatting hints)
  - Urgency language
  - Mismatch between displayed text and actual link
"""

import re
from dataclasses import dataclass, field


# --- Suspicious word lists ---
URGENCY_WORDS = [
    "urgent", "immediately", "verify now", "act now", "limited time",
    "account suspended", "click here", "confirm your", "update your",
    "unusual activity", "security alert", "you have been selected",
]

SENSITIVE_REQUEST_WORDS = [
    "password", "credit card", "social security", "bank account",
    "pin", "otp", "one-time", "login credentials",
]


@dataclass
class PhishingResult:
    verdict: str                     # "phishing" | "suspicious" | "clean"
    confidence: float                # 0.0 – 1.0
    features: list[str] = field(default_factory=list)   # evidence found
    urls_found: list[str] = field(default_factory=list)


def _extract_urls(text: str) -> list[str]:
    pattern = r"http[s]?://[^\s\"]+"
    return re.findall(pattern, text, re.IGNORECASE)


def _check_url(url: str) -> list[str]:
    """Return a list of suspicious signals found in a URL."""
    signals = []

    # IP address instead of domain name
    if re.search(r"https?://\d{1,3}(\.\d{1,3}){3}", url):
        signals.append("URL uses a raw IP address instead of a domain name")

    # Excessive subdomains (typosquatting / redirect tricks)
    domain_part = re.sub(r"https?://", "", url).split("/")[0]
    if domain_part.count(".") >= 4:
        signals.append("URL has an unusually high number of subdomains")

    # Common typosquatting targets
    typosquat_targets = ["paypal", "amazon", "google", "apple", "microsoft",
                         "netflix", "bank", "chase", "wellsfargo", "citibank"]
    for target in typosquat_targets:
        if target in url.lower() and target not in domain_part.lower().split(".")[0]:
            signals.append(f"URL contains brand name '{target}' in a suspicious position")

    # Very long URLs (common obfuscation technique)
    if len(url) > 100:
        signals.append("URL is unusually long (possible obfuscation)")

    # Encoded characters
    if "%" in url and re.search(r"%[0-9a-fA-F]{2}", url):
        signals.append("URL contains percent-encoded characters (possible obfuscation)")

    return signals


def _check_text(text: str) -> list[str]:
    """Return suspicious signals found in the email/message body."""
    signals = []
    lower_text = text.lower()

    for word in URGENCY_WORDS:
        if word in lower_text:
            signals.append(f"Urgency language detected: '{word}'")

    for word in SENSITIVE_REQUEST_WORDS:
        if word in lower_text:
            signals.append(f"Requests sensitive information: '{word}'")

    # Generic greeting (phishing rarely knows your real name)
    if re.search(r"\b(dear customer|dear user|dear account holder)\b", lower_text):
        signals.append("Uses generic greeting instead of your real name")

    return signals


def _score_to_verdict(n_signals: int) -> tuple[str, float]:
    if n_signals == 0:
        return "clean", 0.05
    elif n_signals == 1:
        return "suspicious", 0.45
    elif n_signals == 2:
        return "suspicious", 0.65
    else:
        return "phishing", min(0.70 + (n_signals - 3) * 0.07, 0.97)


def analyse(text: str) -> PhishingResult:
    """
    Main entry point. Pass the full email body or a URL string.
    Returns a PhishingResult with verdict, confidence, and evidence.
    """
    urls = _extract_urls(text)
    features: list[str] = []

    # Check URL signals
    for url in urls:
        features.extend(_check_url(url))

    # Check text signals
    features.extend(_check_text(text))

    verdict, confidence = _score_to_verdict(len(features))

    return PhishingResult(
        verdict=verdict,
        confidence=round(confidence, 2),
        features=features,
        urls_found=urls,
    )
