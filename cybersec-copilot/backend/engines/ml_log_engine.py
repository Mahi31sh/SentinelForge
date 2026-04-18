"""
Phase 4 — ML Log Anomaly Detection
=====================================
Adds Isolation Forest anomaly scoring on top of the rule-based log engine.
Detects unusual patterns that rules alone can't catch:
  - Abnormal login times (3 AM spike)
  - Unusual source IP volume distribution
  - Login velocity anomalies
  - Mixed-signal attacks (slow brute force that avoids thresholds)

Install:
  pip install scikit-learn pandas numpy joblib

The rule engine runs first — ML adds an anomaly score on top.
Both outputs are merged into the final explanation.
"""

from __future__ import annotations
import re
import json
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "models" / "log_anomaly_model.pkl"

# ── Lazy model loading ────────────────────────────────────────────────────────

_anomaly_model = None
_model_tried   = False


def _load_model():
    global _anomaly_model, _model_tried
    if _model_tried:
        return _anomaly_model
    _model_tried = True
    if MODEL_PATH.exists():
        try:
            import joblib
            _anomaly_model = joblib.load(MODEL_PATH)
            print("[ML-Log] Anomaly model loaded.")
        except Exception as e:
            print(f"[ML-Log] Could not load model: {e}")
    return _anomaly_model


# ── Log parsing ───────────────────────────────────────────────────────────────

_FAILED_RE  = re.compile(r"Failed password.*?from (?P<ip>\d{1,3}(?:\.\d{1,3}){3})", re.IGNORECASE)
_SUCCESS_RE = re.compile(r"Accepted (?:password|publickey) for (?P<user>\S+) from (?P<ip>\d{1,3}(?:\.\d{1,3}){3})", re.IGNORECASE)
_TIME_RE    = re.compile(r"\b(\d{1,2}):(\d{2}):\d{2}\b")
_SUDO_RE    = re.compile(r"sudo.*COMMAND=(?P<cmd>.+)", re.IGNORECASE)


def _parse_events(log_text: str) -> dict:
    failures   = defaultdict(int)    # ip -> count
    successes  = defaultdict(list)   # ip -> [users]
    hours      = []
    sudo_cmds  = []
    all_ips    = set()

    for line in log_text.splitlines():
        m = _FAILED_RE.search(line)
        if m:
            ip = m.group("ip")
            failures[ip] += 1
            all_ips.add(ip)

        m = _SUCCESS_RE.search(line)
        if m:
            successes[m.group("ip")].append(m.group("user"))
            all_ips.add(m.group("ip"))

        m = _TIME_RE.search(line)
        if m:
            hours.append(int(m.group(1)))

        m = _SUDO_RE.search(line)
        if m:
            sudo_cmds.append(m.group("cmd").strip())

    return {
        "failures": dict(failures),
        "successes": dict(successes),
        "hours": hours,
        "sudo_cmds": sudo_cmds,
        "all_ips": list(all_ips),
        "total_lines": len(log_text.splitlines()),
    }


# ── Feature vector for anomaly scoring ────────────────────────────────────────

def _build_feature_vector(events: dict) -> list[float]:
    """
    Build a 12-dimensional feature vector for anomaly detection.
    Isolation Forest will score this against its training distribution.
    """
    failures  = events["failures"]
    successes = events["successes"]
    hours     = events["hours"]

    total_failures = sum(failures.values())
    n_ips          = len(failures)
    max_from_one   = max(failures.values()) if failures else 0

    # Concentration: if one IP causes >80% of failures → suspicious
    concentration = (max_from_one / total_failures) if total_failures > 0 else 0

    # After-hours ratio (23:00–05:00)
    after_hours = sum(1 for h in hours if h >= 23 or h <= 5)
    after_ratio = after_hours / len(hours) if hours else 0

    # Avg failures per IP
    avg_per_ip = total_failures / n_ips if n_ips > 0 else 0

    # Success after failures (compromise indicator)
    compromise_ips = sum(1 for ip in successes if ip in failures)

    # Distributed attack (many IPs, lower individual counts)
    distributed_score = n_ips / (max_from_one + 1)

    # Sudo usage
    has_sudo = float(len(events["sudo_cmds"]) > 0)
    dangerous_sudo = float(any(
        kw in cmd for cmd in events["sudo_cmds"]
        for kw in ["/bin/bash", "/bin/sh", "wget", "curl", "nc ", "chmod 777"]
    ))

    return [
        float(total_failures),
        float(n_ips),
        float(max_from_one),
        concentration,
        after_ratio,
        avg_per_ip,
        float(compromise_ips),
        distributed_score,
        float(len(successes)),
        float(len(events["sudo_cmds"])),
        has_sudo,
        dangerous_sudo,
    ]


# ── Anomaly scoring without pre-trained model ─────────────────────────────────

def _heuristic_anomaly_score(features: list[float]) -> tuple[float, str]:
    """
    Rule-of-thumb anomaly score when no trained model exists.
    Returns (score 0-1, reason).
    score > 0.6 = anomalous
    """
    (total_fail, n_ips, max_one, concentration, after_ratio,
     avg_per_ip, compromise, distributed, n_success, n_sudo,
     has_sudo, danger_sudo) = features

    score = 0.0
    reasons = []

    if total_fail > 20:
        score += 0.25
        reasons.append(f"High failure volume ({int(total_fail)} attempts)")
    if concentration > 0.8 and total_fail > 10:
        score += 0.2
        reasons.append("Single IP responsible for most failures (focused attack)")
    if after_ratio > 0.6:
        score += 0.2
        reasons.append(f"{int(after_ratio*100)}% of events are after-hours (23:00–05:00)")
    if compromise > 0:
        score += 0.3
        reasons.append(f"{int(compromise)} IP(s) succeeded after prior failures")
    if distributed > 5 and total_fail > 30:
        score += 0.15
        reasons.append("Many IPs each with low attempts (distributed/credential-stuffing pattern)")
    if danger_sudo:
        score += 0.25
        reasons.append("Dangerous sudo command executed (possible privilege escalation)")

    score = min(score, 1.0)
    level = "high" if score >= 0.6 else "medium" if score >= 0.3 else "low"
    return score, level, reasons


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    anomaly_score: float        # 0.0 – 1.0  (higher = more anomalous)
    anomaly_level: str          # "low" | "medium" | "high"
    anomaly_reasons: list[str]
    method: str                 # "isolation_forest" | "heuristic"
    events: dict = field(default_factory=dict, repr=False)


# ── Public API ────────────────────────────────────────────────────────────────

def analyse_anomaly(log_text: str) -> AnomalyResult:
    """
    Score log text for anomalous patterns.
    Combines with rule-based findings from log_engine.py in chat.py.
    """
    events  = _parse_events(log_text)
    feats   = _build_feature_vector(events)
    model   = _load_model()

    if model is not None:
        import numpy as np
        # Isolation Forest returns -1 (anomaly) or 1 (normal)
        # decision_function gives continuous score (more negative = more anomalous)
        decision = model.decision_function([feats])[0]
        # Normalise to 0-1 (0=normal, 1=highly anomalous)
        score  = max(0.0, min(1.0, 0.5 - decision * 0.5))
        level  = "high" if score >= 0.65 else "medium" if score >= 0.35 else "low"
        _, _, reasons = _heuristic_anomaly_score(feats)   # still get reasons
        method = "isolation_forest"
    else:
        score, level, reasons = _heuristic_anomaly_score(feats)
        method = "heuristic"

    return AnomalyResult(
        anomaly_score=round(score, 3),
        anomaly_level=level,
        anomaly_reasons=reasons,
        method=method,
        events=events,
    )


def train_isolation_forest(log_samples: list[str] | None = None):
    """
    Train an Isolation Forest on normal log samples.

    In production: collect ~1000 lines of clean auth.log from your server.
    Here we generate synthetic clean data so you can see the pipeline.

    Run:  python scripts/train_log_model.py
    """
    import numpy as np
    import joblib
    from sklearn.ensemble import IsolationForest

    if log_samples:
        # Use provided real logs
        feature_matrix = []
        for log in log_samples:
            events = _parse_events(log)
            feature_matrix.append(_build_feature_vector(events))
        X = np.array(feature_matrix)
    else:
        # Synthetic normal baseline: low failure counts, daytime hours
        rng = np.random.default_rng(42)
        n   = 500
        X   = np.column_stack([
            rng.integers(0, 5, n).astype(float),      # total_failures: mostly 0-5
            rng.integers(0, 3, n).astype(float),       # n_ips: 0-2
            rng.integers(0, 3, n).astype(float),       # max_from_one: 0-2
            rng.uniform(0, 0.3, n),                    # concentration: low
            rng.uniform(0, 0.15, n),                   # after_ratio: mostly daytime
            rng.uniform(0, 2, n),                      # avg_per_ip: low
            np.zeros(n),                               # compromise: 0
            rng.uniform(0, 2, n),                      # distributed: low
            rng.integers(0, 5, n).astype(float),       # n_success: low
            np.zeros(n),                               # n_sudo: 0
            np.zeros(n),                               # has_sudo: 0
            np.zeros(n),                               # danger_sudo: 0
        ])

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,   # expect ~5% anomalies in normal logs
        random_state=42,
    )
    model.fit(X)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[ML-Log] Isolation Forest trained on {len(X)} samples.")
    print(f"[ML-Log] Model saved to {MODEL_PATH}")
    return model
