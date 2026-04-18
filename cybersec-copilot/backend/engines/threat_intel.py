import requests
import os
from dotenv import load_dotenv


load_dotenv()

import base64

def check_url_virustotal(url: str):
    vt_api_key = os.getenv("VIRUSTOTAL_API_KEY")
    if not vt_api_key:
        return None

    headers = {"x-apikey": vt_api_key}

    try:
        # Encode URL for VT
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")

        response = requests.get(
            f"https://www.virustotal.com/api/v3/urls/{url_id}",
            headers=headers,
            timeout=12,
        )

        data = response.json()

        stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})

        return {
            "malicious": stats.get("malicious", 0),
            "suspicious": stats.get("suspicious", 0),
            "harmless": stats.get("harmless", 0)
        }

    except Exception as e:
        print("VT ERROR:", e)
        return None


def check_ip_abuse(ip: str):
    abuse_api_key = os.getenv("ABUSEIPDB_API_KEY")
    if not abuse_api_key:
        return None

    headers = {
        "Key": abuse_api_key,
        "Accept": "application/json"
    }

    params = {
        "ipAddress": ip,
        "maxAgeInDays": 90
    }

    try:
        response = requests.get(
            "https://api.abuseipdb.com/api/v2/check",
            headers=headers,
            params=params,
            timeout=12,
        )
        return response.json()
    except Exception as e:
        print("ABUSE ERROR:", e)
        return None