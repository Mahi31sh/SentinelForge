"""
Phase 2 — Seed the RAG knowledge base.

Usage:
  python scripts/seed_knowledge.py            # seed once
  python scripts/seed_knowledge.py --force    # re-index everything

How to add your own knowledge:
  1. Download OWASP content, CVE descriptions, or paste any security text
  2. Save as .txt files inside  data/raw/
  3. Run this script again with --force

Free knowledge sources:
  - OWASP Cheat Sheet Series : https://cheatsheetseries.owasp.org/
  - NVD CVE feeds            : https://nvd.nist.gov/vuln/data-feeds
  - MITRE ATT&CK             : https://attack.mitre.org/
"""

import sys
from pathlib import Path

# Make sure backend package is importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.engines.knowledge_engine import seed, collection_size

force = "--force" in sys.argv

print("=" * 52)
print("  SentinelForge — RAG Knowledge Base Seeder")
print("=" * 52)

n = seed(force=force)

print(f"\nCollection now has {collection_size()} total chunks.")
print("Knowledge base is ready. Restart the API server.")
