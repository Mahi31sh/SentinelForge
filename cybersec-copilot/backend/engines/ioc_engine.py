"""
IOC Enrichment Engine  —  Phase 3
====================================
Queries real threat-intelligence APIs to enrich Indicators of Compromise (IOCs).

Supported providers (all have FREE tiers — no credit card needed):
  - VirusTotal   : URLs, domains, IPs, file hashes  (500 req/day free)
  - AbuseIPDB    : IP reputation                    (1000 req/day free)
  - Shodan       : Host/port intelligence           (100 req/month free)

Setup:
  1. Get free API keys:
       VirusTotal : https://www.virustotal.com/gui/join-us
       AbuseIPDB  : https://www.abuseipdb.com/register
       Shodan     : https://account.shodan.io/register

  2. Add them to your .env file:
       VIRUSTOTAL_API_KEY=your_key_here
       ABUSEIPDB_API_KEY=your_key_here
       SHODAN_API_KEY=your_key_here

  3. Install requests: pip install requests python-dotenv

The engine degrades gracefully — if a key is missing, that provider is skipped.
"""

from __future__ import annotations
import os
import re
import base64
import hashlib
import ipaddress
from dataclasses import dataclass, field
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

VT_KEY      = os.getenv("VIRUSTOTAL_API_KEY", "")
ABUSE_KEY   = os.getenv("ABUSEIPDB_API_KEY", "")
SHODAN_KEY  = os.getenv("SHODAN_API_KEY", "")

TIMEOUT = 8   # seconds per API call


# ── Result model ─────────────────────────────────────────────────────────────

@dataclass
class IOCResult:
    indicator: str
    indicator_type: str          # "url" | "ip" | "domain" | "hash"
    provider: str
    status: str                  # "malicious" | "suspicious" | "clean" | "unknown"
    score: int | None            # provider-specific score (e.g. VT detection count)
    summary: str
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class EnrichmentReport:
    results: list[IOCResult]
    indicators_checked: list[str]
    errors: list[str]

    @property
    def highest_risk(self) -> str:
        order = {"malicious": 3, "suspicious": 2, "clean": 1, "unknown": 0}
        if not self.results:
            return "unknown"
        return max(self.results, key=lambda r: order.get(r.status, 0)).status


# ── IOC extraction ────────────────────────────────────────────────────────────

_URL_RE  = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_IP_RE   = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_HASH_RE = re.compile(r"\b[0-9a-fA-F]{32,64}\b")    # MD5/SHA1/SHA256
_DOM_RE  = re.compile(r"\b(?:[a-zA-Z0-9-]{1,63}\.)+(?:com|net|org|io|xyz|tk|ml|ga|cf|gq|info|biz|co)\b", re.IGNORECASE)


def extract_iocs(text: str) -> dict[str, list[str]]:
    """Extract URLs, IPs, hashes, and domains from free text."""
    urls    = list(dict.fromkeys(_URL_RE.findall(text)))
    ips     = [ip for ip in dict.fromkeys(_IP_RE.findall(text))
               if _is_public_ip(ip)]
    hashes  = list(dict.fromkeys(_HASH_RE.findall(text)))
    # Domains found in text that are NOT already in a URL
    url_bodies = " ".join(urls)
    domains = [d for d in dict.fromkeys(_DOM_RE.findall(text))
               if d not in url_bodies]
    return {"urls": urls, "ips": ips, "hashes": hashes, "domains": domains}


def _is_public_ip(ip_str: str) -> bool:
    try:
        return not ipaddress.ip_address(ip_str).is_private
    except ValueError:
        return False


# ── VirusTotal ────────────────────────────────────────────────────────────────

def _vt_headers() -> dict:
    return {"x-apikey": VT_KEY, "Accept": "application/json"}


def _vt_check_url(url: str) -> IOCResult:
    """Check a URL against VirusTotal (70+ engines)."""
    encoded = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    try:
        r = requests.get(
            f"https://www.virustotal.com/api/v3/urls/{encoded}",
            headers=_vt_headers(),
            timeout=TIMEOUT,
        )
        if r.status_code == 404:
            # URL not in VT cache — submit it for analysis
            submit = requests.post(
                "https://www.virustotal.com/api/v3/urls",
                headers=_vt_headers(),
                data={"url": url},
                timeout=TIMEOUT,
            )
            return IOCResult(
                indicator=url, indicator_type="url", provider="VirusTotal",
                status="unknown", score=None,
                summary="URL submitted for analysis. Check back in a few minutes.",
            )

        r.raise_for_status()
        data   = r.json()
        stats  = data["data"]["attributes"]["last_analysis_stats"]
        mal    = stats.get("malicious", 0)
        sus    = stats.get("suspicious", 0)
        total  = sum(stats.values()) or 1
        status = "malicious" if mal >= 3 else "suspicious" if (mal + sus) >= 1 else "clean"
        return IOCResult(
            indicator=url, indicator_type="url", provider="VirusTotal",
            status=status, score=mal,
            summary=f"{mal}/{total} engines flagged as malicious, {sus} suspicious.",
            raw=stats,
        )
    except requests.RequestException as e:
        return IOCResult(indicator=url, indicator_type="url", provider="VirusTotal",
                         status="unknown", score=None, summary=f"Request failed: {e}")


def _vt_check_ip(ip: str) -> IOCResult:
    """Check an IP address against VirusTotal."""
    try:
        r = requests.get(
            f"https://www.virustotal.com/api/v3/ip_addresses/{ip}",
            headers=_vt_headers(),
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data  = r.json()
        stats = data["data"]["attributes"]["last_analysis_stats"]
        mal   = stats.get("malicious", 0)
        sus   = stats.get("suspicious", 0)
        total = sum(stats.values()) or 1
        country = data["data"]["attributes"].get("country", "unknown")
        status = "malicious" if mal >= 3 else "suspicious" if (mal + sus) >= 1 else "clean"
        return IOCResult(
            indicator=ip, indicator_type="ip", provider="VirusTotal",
            status=status, score=mal,
            summary=f"{mal}/{total} engines flagged. Country: {country}.",
            raw=stats,
        )
    except requests.RequestException as e:
        return IOCResult(indicator=ip, indicator_type="ip", provider="VirusTotal",
                         status="unknown", score=None, summary=f"Request failed: {e}")


def _vt_check_hash(hash_str: str) -> IOCResult:
    """Check a file hash (MD5/SHA1/SHA256) against VirusTotal."""
    try:
        r = requests.get(
            f"https://www.virustotal.com/api/v3/files/{hash_str}",
            headers=_vt_headers(),
            timeout=TIMEOUT,
        )
        if r.status_code == 404:
            return IOCResult(indicator=hash_str, indicator_type="hash",
                             provider="VirusTotal", status="unknown", score=None,
                             summary="Hash not found in VirusTotal database.")
        r.raise_for_status()
        data  = r.json()
        stats = data["data"]["attributes"]["last_analysis_stats"]
        mal   = stats.get("malicious", 0)
        total = sum(stats.values()) or 1
        name  = data["data"]["attributes"].get("meaningful_name", "unknown file")
        status = "malicious" if mal >= 3 else "suspicious" if mal >= 1 else "clean"
        return IOCResult(
            indicator=hash_str, indicator_type="hash", provider="VirusTotal",
            status=status, score=mal,
            summary=f"File: {name}. {mal}/{total} engines flagged as malicious.",
            raw=stats,
        )
    except requests.RequestException as e:
        return IOCResult(indicator=hash_str, indicator_type="hash",
                         provider="VirusTotal", status="unknown", score=None,
                         summary=f"Request failed: {e}")


# ── AbuseIPDB ─────────────────────────────────────────────────────────────────

def _abuse_check_ip(ip: str) -> IOCResult:
    """Check IP reputation against AbuseIPDB."""
    try:
        r = requests.get(
            "https://api.abuseipdb.com/api/v2/check",
            headers={"Key": ABUSE_KEY, "Accept": "application/json"},
            params={"ipAddress": ip, "maxAgeInDays": 90, "verbose": True},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data   = r.json()["data"]
        score  = data.get("abuseConfidenceScore", 0)
        usage  = data.get("usageType", "unknown")
        isp    = data.get("isp", "unknown ISP")
        reports= data.get("totalReports", 0)
        country= data.get("countryCode", "?")
        status = "malicious" if score >= 75 else "suspicious" if score >= 25 else "clean"
        return IOCResult(
            indicator=ip, indicator_type="ip", provider="AbuseIPDB",
            status=status, score=score,
            summary=(f"Abuse confidence: {score}%. {reports} reports. "
                     f"ISP: {isp}. Usage: {usage}. Country: {country}."),
            raw=data,
        )
    except requests.RequestException as e:
        return IOCResult(indicator=ip, indicator_type="ip", provider="AbuseIPDB",
                         status="unknown", score=None, summary=f"Request failed: {e}")


# ── Shodan ────────────────────────────────────────────────────────────────────

def _shodan_check_ip(ip: str) -> IOCResult:
    """Look up host info from Shodan (open ports, services, vulns)."""
    try:
        r = requests.get(
            f"https://api.shodan.io/shodan/host/{ip}",
            params={"key": SHODAN_KEY},
            timeout=TIMEOUT,
        )
        if r.status_code == 404:
            return IOCResult(indicator=ip, indicator_type="ip", provider="Shodan",
                             status="unknown", score=None,
                             summary="Host not indexed by Shodan.")
        r.raise_for_status()
        data  = r.json()
        ports = data.get("ports", [])
        vulns = list(data.get("vulns", {}).keys())
        org   = data.get("org", "unknown org")
        os_   = data.get("os", "unknown OS")
        status = "malicious" if vulns else "suspicious" if len(ports) > 10 else "clean"
        vuln_str = (f" Known CVEs: {', '.join(vulns[:3])}{'...' if len(vulns)>3 else ''}."
                    if vulns else " No known CVEs.")
        return IOCResult(
            indicator=ip, indicator_type="ip", provider="Shodan",
            status=status, score=len(vulns),
            summary=f"Org: {org}. OS: {os_}. Open ports: {ports[:8]}.{vuln_str}",
            raw={"ports": ports, "vulns": vulns},
        )
    except requests.RequestException as e:
        return IOCResult(indicator=ip, indicator_type="ip", provider="Shodan",
                         status="unknown", score=None, summary=f"Request failed: {e}")


# ── Main enrichment entry point ───────────────────────────────────────────────

def enrich(text: str, max_iocs: int = 3) -> EnrichmentReport:
    """
    Extract IOCs from text and query all configured providers.

    Args:
        text     : Raw user input (email body, log paste, or URL)
        max_iocs : Maximum number of unique IOCs to check (rate-limit safety)

    Returns:
        EnrichmentReport with all results and any errors encountered.
    """
    iocs   = extract_iocs(text)
    results: list[IOCResult] = []
    errors: list[str] = []
    checked: list[str] = []

    count = 0

    # ── URLs via VirusTotal ───────────────────────────────────────────────
    if VT_KEY:
        for url in iocs["urls"][:max_iocs]:
            if count >= max_iocs:
                break
            results.append(_vt_check_url(url))
            checked.append(url)
            count += 1
    elif iocs["urls"]:
        errors.append("VIRUSTOTAL_API_KEY not set — URL checks skipped.")

    # ── IPs via AbuseIPDB + Shodan + VirusTotal ───────────────────────────
    for ip in iocs["ips"][:max_iocs]:
        if count >= max_iocs:
            break
        if ABUSE_KEY:
            results.append(_abuse_check_ip(ip))
        if SHODAN_KEY:
            results.append(_shodan_check_ip(ip))
        if VT_KEY:
            results.append(_vt_check_ip(ip))
        if not (ABUSE_KEY or SHODAN_KEY or VT_KEY):
            errors.append("No API keys set — IP checks skipped.")
            break
        checked.append(ip)
        count += 1

    # ── File hashes via VirusTotal ────────────────────────────────────────
    if VT_KEY:
        for h in iocs["hashes"][:max_iocs]:
            if count >= max_iocs:
                break
            results.append(_vt_check_hash(h))
            checked.append(h)
            count += 1
    elif iocs["hashes"]:
        errors.append("VIRUSTOTAL_API_KEY not set — hash checks skipped.")

    # ── No IOCs found ─────────────────────────────────────────────────────
    if not checked:
        errors.append("No public IOCs (URLs, IPs, hashes) found in the input.")

    return EnrichmentReport(results=results, indicators_checked=checked, errors=errors)
