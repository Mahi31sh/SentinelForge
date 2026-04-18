"""
Phase 4 — ML Phishing Engine
==============================
Replaces rule-based scoring with a trained RandomForest classifier.

Strategy:
  - If ML model exists (models/phishing_classifier.pkl) → use ML prediction
  - If ML model missing → fall back to Phase 1 rule-based engine
  - Always combine ML confidence with rule-based evidence for the explanation

This means the chatbot NEVER breaks — it degrades gracefully from
ML → rules if the model file is not found.

Install:
  pip install scikit-learn pandas numpy joblib

Train:
  python scripts/train_phishing_model.py
"""

from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "models" / "phishing_classifier.pkl"
META_PATH  = ROOT / "models" / "phishing_classifier_meta.json"

# ── Lazy model loading ────────────────────────────────────────────────────────

_model       = None
_meta        = None
_model_tried = False


def _load_model():
    global _model, _meta, _model_tried
    if _model_tried:
        return _model
    _model_tried = True
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib
        _model = joblib.load(MODEL_PATH)
        if META_PATH.exists():
            _meta = json.loads(META_PATH.read_text())
        print(f"[ML] Phishing model loaded. AUC-ROC: {_meta.get('auc_roc','?') if _meta else '?'}")
    except Exception as e:
        print(f"[ML] Could not load model: {e}")
        _model = None
    return _model


def model_available() -> bool:
    return _load_model() is not None


# ── Feature names (must match training script exactly) ───────────────────────

FEATURE_NAMES = [
    "url_length", "domain_length", "path_length", "has_https", "has_http_only",
    "has_ip_address", "subdomain_count", "has_deep_subdomain", "at_sign_count",
    "dash_in_domain", "dot_count", "digit_in_domain", "special_char_count",
    "hex_encoded_chars", "brand_in_domain", "brand_in_path", "suspicious_tld",
    "trusted_tld", "domain_entropy", "suspicious_words_in_url",
]

SUSPICIOUS_WORDS = [
    "verify", "account", "secure", "update", "login", "banking", "confirm",
    "password", "credit", "urgent", "suspended", "click", "limited", "free",
    "winner", "congratulations", "prize", "offer", "alert", "warning",
]

BRAND_NAMES = [
    "paypal", "amazon", "google", "apple", "microsoft", "netflix", "facebook",
    "instagram", "twitter", "bank", "chase", "wellsfargo", "citibank",
    "linkedin", "dropbox", "adobe", "yahoo", "outlook", "office365",
]

TRUSTED_TLDS = {".com", ".org", ".net", ".gov", ".edu", ".co.uk", ".io"}


# ── Feature extraction ────────────────────────────────────────────────────────

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {c: s.count(c) / len(s) for c in set(s)}
    return -sum(p * math.log2(p) for p in freq.values())


def extract_features(url: str) -> dict:
    url_lower = url.lower()
    has_https = int(url_lower.startswith("https://"))

    try:
        domain_part = re.sub(r"https?://", "", url_lower).split("/")[0]
        path_part   = "/" + "/".join(re.sub(r"https?://", "", url_lower).split("/")[1:])
    except Exception:
        domain_part, path_part = url_lower, ""

    tld = "." + domain_part.split(".")[-1] if "." in domain_part else ""

    return {
        "url_length":              len(url),
        "domain_length":           len(domain_part),
        "path_length":             len(path_part),
        "has_https":               has_https,
        "has_http_only":           int(url_lower.startswith("http://") and not has_https),
        "has_ip_address":          int(bool(re.match(r"\d{1,3}(\.\d{1,3}){3}", domain_part))),
        "subdomain_count":         max(0, domain_part.count(".") - 1),
        "has_deep_subdomain":      int(max(0, domain_part.count(".") - 1) >= 3),
        "at_sign_count":           url.count("@"),
        "dash_in_domain":          domain_part.count("-"),
        "dot_count":               url.count("."),
        "digit_in_domain":         sum(c.isdigit() for c in domain_part),
        "special_char_count":      sum(1 for c in url if c in "%_=?&#"),
        "hex_encoded_chars":       len(re.findall(r"%[0-9a-fA-F]{2}", url)),
        "brand_in_domain":         int(any(b in domain_part for b in BRAND_NAMES)),
        "brand_in_path":           int(any(b in path_part for b in BRAND_NAMES)
                                       and not any(b in domain_part for b in BRAND_NAMES)),
        "suspicious_tld":          int(tld in {".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".click"}),
        "trusted_tld":             int(tld in TRUSTED_TLDS),
        "domain_entropy":          round(_entropy(domain_part.replace(".", "")), 4),
        "suspicious_words_in_url": sum(w in url_lower for w in SUSPICIOUS_WORDS),
    }


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class MLPhishingResult:
    verdict: str              # "phishing" | "suspicious" | "clean"
    confidence: float         # 0.0 – 1.0
    features: list[str]       # human-readable evidence
    urls_found: list[str]
    method: str               # "ml" | "rules"
    top_feature_contributions: list[str] = field(default_factory=list)


# ── Rule-based fallback (Phase 1 logic, kept here for graceful degradation) ──

_URGENCY = ["urgent", "immediately", "verify now", "act now", "limited time",
            "account suspended", "click here", "confirm your", "update your",
            "unusual activity", "security alert"]
_SENSITIVE = ["password", "credit card", "social security", "bank account",
              "pin", "otp", "one-time", "login credentials"]

_URL_RE = re.compile(r"http[s]?://[^\s\"]+")


def _rule_based_analyse(text: str) -> MLPhishingResult:
    """Phase 1 rule engine — used when no ML model is available."""
    urls = _URL_RE.findall(text)
    evidence = []
    lower = text.lower()

    for url in urls:
        f = extract_features(url)
        if f["has_ip_address"]:
            evidence.append("URL uses a raw IP address instead of a domain name")
        if f["subdomain_count"] >= 4:
            evidence.append("URL has an unusually high number of subdomains")
        if f["suspicious_tld"]:
            evidence.append(f"URL uses a suspicious free TLD (high phishing association)")
        if f["brand_in_domain"] and not f["trusted_tld"]:
            evidence.append("Brand name appears in a suspicious domain position")
        if f["url_length"] > 100:
            evidence.append("URL is unusually long (common obfuscation technique)")
        if f["hex_encoded_chars"] > 3:
            evidence.append("URL contains percent-encoded characters (possible obfuscation)")

    for word in _URGENCY:
        if word in lower:
            evidence.append(f"Urgency language: '{word}'")
    for word in _SENSITIVE:
        if word in lower:
            evidence.append(f"Requests sensitive information: '{word}'")
    if re.search(r"\b(dear customer|dear user|dear account holder)\b", lower):
        evidence.append("Uses generic greeting instead of your real name")

    n = len(evidence)
    if n == 0:
        verdict, conf = "clean", 0.05
    elif n == 1:
        verdict, conf = "suspicious", 0.45
    elif n == 2:
        verdict, conf = "suspicious", 0.65
    else:
        verdict, conf = "phishing", min(0.70 + (n - 3) * 0.07, 0.97)

    return MLPhishingResult(
        verdict=verdict, confidence=round(conf, 2),
        features=evidence, urls_found=urls, method="rules",
    )


# ── ML-based analysis ─────────────────────────────────────────────────────────

def _ml_analyse(text: str) -> MLPhishingResult:
    """ML prediction. Called only when model is loaded."""
    import numpy as np

    urls = _URL_RE.findall(text)
    model = _load_model()

    # If no URLs found, still run rule-based text analysis
    if not urls:
        result = _rule_based_analyse(text)
        result.method = "rules (no URL found)"
        return result

    # Score every URL and take the worst case
    worst_proba = 0.0
    worst_url   = urls[0]
    all_features = []

    for url in urls:
        feat_dict = extract_features(url)
        feat_vec  = np.array([[feat_dict[f] for f in FEATURE_NAMES]])
        proba     = model.predict_proba(feat_vec)[0][1]   # prob of phishing
        if proba > worst_proba:
            worst_proba = proba
            worst_url   = url
            all_features = feat_dict

    # Verdict thresholds
    if worst_proba >= 0.65:
        verdict = "phishing"
    elif worst_proba >= 0.35:
        verdict = "suspicious"
    else:
        verdict = "clean"

    # Generate human-readable evidence from top contributing features
    evidence = _features_to_evidence(all_features, worst_url)

    # Feature contributions (show top model drivers)
    contributions = []
    if _meta and "top_features" in _meta:
        for feat_name in _meta["top_features"][:3]:
            val = all_features.get(feat_name, 0)
            if val:
                contributions.append(f"Model signal: {feat_name.replace('_',' ')} = {val}")

    return MLPhishingResult(
        verdict=verdict,
        confidence=round(float(worst_proba), 2),
        features=evidence,
        urls_found=urls,
        method="ml",
        top_feature_contributions=contributions,
    )


def _features_to_evidence(feat: dict, url: str) -> list[str]:
    """Turn numeric features back into readable evidence strings."""
    ev = []
    if feat.get("has_ip_address"):
        ev.append("URL uses a raw IP address instead of a registered domain")
    if feat.get("has_deep_subdomain"):
        ev.append(f"URL has {feat['subdomain_count']} subdomain levels (suspicious depth)")
    if feat.get("suspicious_tld"):
        tld = url.split(".")[-1].split("/")[0]
        ev.append(f"Domain uses high-risk free TLD: .{tld}")
    if feat.get("brand_in_domain") and not feat.get("trusted_tld"):
        ev.append("Brand name in domain but hosted on untrusted TLD (typosquatting pattern)")
    if feat.get("url_length", 0) > 100:
        ev.append(f"URL length {feat['url_length']} chars — abnormally long (obfuscation risk)")
    if feat.get("hex_encoded_chars", 0) > 2:
        ev.append("URL contains encoded characters (%xx) — common evasion technique")
    if feat.get("at_sign_count", 0) > 0:
        ev.append("@ symbol in URL — browser ignores everything before it (redirect trick)")
    if feat.get("dash_in_domain", 0) >= 3:
        ev.append(f"{feat['dash_in_domain']} dashes in domain — phishing domains commonly use hyphens")
    if feat.get("domain_entropy", 0) > 3.8:
        ev.append("Domain name has high character entropy — looks randomly generated")
    if feat.get("suspicious_words_in_url", 0) >= 2:
        ev.append(f"{feat['suspicious_words_in_url']} suspicious security-related words in URL")
    if not feat.get("has_https"):
        ev.append("URL uses HTTP (not HTTPS) — no transport encryption")
    return ev or ["No strong phishing signals detected in URL structure"]


# ── Public API ────────────────────────────────────────────────────────────────

def analyse(text: str) -> MLPhishingResult:
    """
    Main entry point. Automatically uses ML if model is trained,
    otherwise falls back to rule-based detection.
    """
    if model_available():
        return _ml_analyse(text)
    return _rule_based_analyse(text)
