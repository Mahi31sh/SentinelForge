"""
Phase 4 — Train the Log Anomaly Model (Isolation Forest)
=========================================================
Run:
  python scripts/train_log_model.py

This trains on synthetic "normal" log behaviour.
For a better model: collect real clean auth.log from your server
and pass lines to train_isolation_forest(log_samples=[...]).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.engines.ml_log_engine import train_isolation_forest

print("=" * 52)
print("  SentinelForge — Phase 4 Log Anomaly Model")
print("=" * 52)
print()

model = train_isolation_forest()

print("\n✓ Log anomaly model trained.")
print("  Restart the API server to activate ML log analysis.")
