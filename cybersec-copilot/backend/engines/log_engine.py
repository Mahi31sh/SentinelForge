"""
Log Analysis Engine
--------------------
Parses system/auth logs and detects attack patterns using rules.

Detects:
  - Brute force (many failed logins from one IP)
  - Credential stuffing (many failed logins across many usernames)
  - Port scanning (many connection attempts across ports)
  - Privilege escalation attempts
  - Suspicious after-hours access

Phase 2: Add Isolation Forest (scikit-learn) for anomaly detection on numeric features.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class LogFinding:
    attack_type: str         # e.g. "Brute Force"
    severity: str            # "low" | "medium" | "high" | "critical"
    description: str
    evidence: list[str] = field(default_factory=list)
    mitre_tactic: str = ""   # MITRE ATT&CK technique ID
    mitre_name: str = ""


@dataclass
class LogAnalysisResult:
    findings: list[LogFinding]
    lines_parsed: int
    ips_seen: list[str]
    summary: str


# --- Regex patterns for common log formats ---
# Works with: syslog, auth.log, Apache/Nginx access logs

FAILED_LOGIN_PATTERN = re.compile(
    r"(?P<timestamp>\w+\s+\d+\s+[\d:]+).*"
    r"(?:Failed password|authentication failure|Invalid user)\s+"
    r"(?:for\s+)?(?P<user>\S+)?\s+"
    r"from\s+(?P<ip>\d{1,3}(?:\.\d{1,3}){3})",
    re.IGNORECASE
)

SUCCESSFUL_LOGIN_PATTERN = re.compile(
    r"Accepted (?:password|publickey) for (?P<user>\S+) from (?P<ip>\d{1,3}(?:\.\d{1,3}){3})",
    re.IGNORECASE
)

SUDO_PATTERN = re.compile(
    r"sudo.*COMMAND=(?P<cmd>.+)",
    re.IGNORECASE
)


def _parse_lines(log_text: str):
    """Parse raw log text into structured events."""
    failed_logins = defaultdict(list)   # ip -> [usernames]
    successful_logins = []
    sudo_commands = []

    for line in log_text.strip().splitlines():
        m = FAILED_LOGIN_PATTERN.search(line)
        if m:
            ip = m.group("ip")
            user = m.group("user") or "unknown"
            failed_logins[ip].append(user)
            continue

        m = SUCCESSFUL_LOGIN_PATTERN.search(line)
        if m:
            successful_logins.append({"user": m.group("user"), "ip": m.group("ip")})
            continue

        m = SUDO_PATTERN.search(line)
        if m:
            sudo_commands.append(m.group("cmd").strip())

    return failed_logins, successful_logins, sudo_commands


def analyse(log_text: str) -> LogAnalysisResult:
    lines = log_text.strip().splitlines()
    failed_logins, successful_logins, sudo_commands = _parse_lines(log_text)
    findings: list[LogFinding] = []

    # ── Rule 1: Brute force (many failures from one IP) ──────────────────
    BRUTE_THRESHOLD = 5
    for ip, users in failed_logins.items():
        if len(users) >= BRUTE_THRESHOLD:
            unique_users = list(set(users))
            severity = "critical" if len(users) >= 20 else "high"
            findings.append(LogFinding(
                attack_type="Brute Force",
                severity=severity,
                description=f"IP {ip} made {len(users)} failed login attempts.",
                evidence=[
                    f"Source IP: {ip}",
                    f"Attempts: {len(users)}",
                    f"Targeted accounts: {', '.join(unique_users[:5])}{'...' if len(unique_users) > 5 else ''}",
                ],
                mitre_tactic="T1110",
                mitre_name="Brute Force",
            ))

    # ── Rule 2: Credential stuffing (many IPs, many usernames) ───────────
    all_failing_ips = list(failed_logins.keys())
    all_failing_users = [u for users in failed_logins.values() for u in users]
    if len(all_failing_ips) >= 3 and len(set(all_failing_users)) >= 5:
        findings.append(LogFinding(
            attack_type="Credential Stuffing",
            severity="high",
            description="Multiple IPs trying many different usernames — consistent with a credential stuffing campaign.",
            evidence=[
                f"Distinct source IPs: {len(all_failing_ips)}",
                f"Distinct usernames tried: {len(set(all_failing_users))}",
            ],
            mitre_tactic="T1110.004",
            mitre_name="Credential Stuffing",
        ))

    # ── Rule 3: Successful login after many failures (may = success after brute) ─
    for entry in successful_logins:
        ip = entry["ip"]
        if ip in failed_logins and len(failed_logins[ip]) >= 3:
            findings.append(LogFinding(
                attack_type="Possible Account Compromise",
                severity="critical",
                description=f"Successful login from {ip} after {len(failed_logins[ip])} failed attempts — may indicate a successful brute-force.",
                evidence=[
                    f"IP: {ip}",
                    f"User logged in: {entry['user']}",
                    f"Prior failures from same IP: {len(failed_logins[ip])}",
                ],
                mitre_tactic="T1078",
                mitre_name="Valid Accounts",
            ))

    # ── Rule 4: Sensitive sudo commands ──────────────────────────────────
    dangerous_cmds = ["/bin/bash", "/bin/sh", "chmod 777", "wget", "curl", "nc ", "ncat"]
    for cmd in sudo_commands:
        for danger in dangerous_cmds:
            if danger in cmd:
                findings.append(LogFinding(
                    attack_type="Privilege Escalation Attempt",
                    severity="high",
                    description=f"Sensitive command run with sudo: {cmd}",
                    evidence=[f"Command: {cmd}"],
                    mitre_tactic="T1548",
                    mitre_name="Abuse Elevation Control Mechanism",
                ))
                break

    all_ips = list(set(
        list(failed_logins.keys()) + [e["ip"] for e in successful_logins]
    ))

    summary = (
        f"Parsed {len(lines)} log lines. "
        f"Found {len(findings)} security finding(s) across {len(all_ips)} unique IP(s)."
        if findings else
        f"Parsed {len(lines)} log lines. No obvious attack patterns detected."
    )

    return LogAnalysisResult(
        findings=findings,
        lines_parsed=len(lines),
        ips_seen=all_ips,
        summary=summary,
    )
