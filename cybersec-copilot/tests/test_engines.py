"""
Quick tests — run with: python -m pytest tests/ -v
No API key needed. These test the detection engines only.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.engines.phishing_engine import analyse as phishing_analyse
from backend.engines.log_engine import analyse as log_analyse
from backend.utils.intent_router import detect_intent, Intent


# ── Intent Router Tests ────────────────────────────────────────────────────

def test_intent_phishing_url():
    assert detect_intent("check this link http://paypal-verify.xyz/login") == Intent.PHISHING

def test_intent_phishing_keyword():
    assert detect_intent("is this email phishing?") == Intent.PHISHING

def test_intent_log():
    assert detect_intent("Failed password for root from 192.168.1.5") == Intent.LOG_ANALYSIS

def test_intent_knowledge():
    assert detect_intent("what is SQL injection") == Intent.KNOWLEDGE


# ── Phishing Engine Tests ─────────────────────────────────────────────────

def test_phishing_detects_ip_url():
    result = phishing_analyse("Click here: http://192.168.1.1/login to verify your account urgently")
    assert result.verdict in ("phishing", "suspicious")
    assert result.confidence > 0.4
    assert len(result.features) > 0

def test_phishing_detects_urgency():
    result = phishing_analyse("Dear customer, your account will be suspended. Verify now immediately.")
    assert result.verdict in ("phishing", "suspicious")

def test_phishing_clean_text():
    result = phishing_analyse("Hi Bob, just checking in about the meeting tomorrow.")
    assert result.verdict == "clean"
    assert result.confidence < 0.3

def test_phishing_finds_url():
    result = phishing_analyse("Visit https://malicious-site.com/steal?token=abc")
    assert len(result.urls_found) == 1
    assert "malicious-site.com" in result.urls_found[0]


# ── Log Engine Tests ──────────────────────────────────────────────────────

BRUTE_FORCE_LOGS = """
Jan 15 10:00:01 server sshd[1234]: Failed password for root from 10.0.0.5 port 54321 ssh2
Jan 15 10:00:02 server sshd[1234]: Failed password for admin from 10.0.0.5 port 54322 ssh2
Jan 15 10:00:03 server sshd[1234]: Failed password for user from 10.0.0.5 port 54323 ssh2
Jan 15 10:00:04 server sshd[1234]: Failed password for root from 10.0.0.5 port 54324 ssh2
Jan 15 10:00:05 server sshd[1234]: Failed password for root from 10.0.0.5 port 54325 ssh2
Jan 15 10:00:06 server sshd[1234]: Failed password for ubuntu from 10.0.0.5 port 54326 ssh2
""".strip()

def test_log_detects_brute_force():
    result = log_analyse(BRUTE_FORCE_LOGS)
    assert result.lines_parsed == 6
    brute_findings = [f for f in result.findings if f.attack_type == "Brute Force"]
    assert len(brute_findings) >= 1
    assert brute_findings[0].mitre_tactic == "T1110"

def test_log_clean():
    clean_log = "Jan 15 09:00:01 server sshd[100]: Accepted password for alice from 10.0.0.2 port 443 ssh2"
    result = log_analyse(clean_log)
    assert len(result.findings) == 0

def test_log_compromise_after_brute():
    logs = BRUTE_FORCE_LOGS + "\nJan 15 10:00:10 server sshd[1234]: Accepted password for root from 10.0.0.5 port 54330 ssh2"
    result = log_analyse(logs)
    compromise = [f for f in result.findings if "Compromise" in f.attack_type]
    assert len(compromise) >= 1
    assert compromise[0].severity == "critical"
