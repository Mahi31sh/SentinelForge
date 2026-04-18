# SentinelForge — AI-Driven Threat Intelligence & Response Engine

A multi-engine cybersecurity copilot with NLP intent routing, phishing detection,
log analysis, RAG knowledge retrieval, real threat-intel APIs, and LLM explanations.

---

## Folder Structure

```
cybersec-copilot/
│
├── backend/
│   ├── main.py                   ← FastAPI entrypoint
│   ├── engines/
│   │   ├── phishing_engine.py    ← Phase 1 · Rule-based phishing detection
│   │   ├── log_engine.py         ← Phase 1 · Log analysis + MITRE mapping
│   │   ├── explainer_engine.py   ← Phase 1 · LLM explanation layer (Claude)
│   │   ├── knowledge_engine.py   ← Phase 2 · RAG with ChromaDB
│   │   └── ioc_engine.py         ← Phase 3 · VirusTotal / AbuseIPDB / Shodan
│   ├── routes/
│   │   └── chat.py               ← POST /api/chat  (all phases wired here)
│   └── utils/
│       └── intent_router.py      ← NLP intent detection
│
├── scripts/
│   └── seed_knowledge.py         ← Phase 2 · Index your .txt files into ChromaDB
│
├── frontend/
│   └── index.html                ← Chat UI (open in browser, no build step)
│
├── data/
│   ├── raw/       ← Drop your .txt knowledge files here (Phase 2)
│   └── chroma/    ← ChromaDB auto-creates this (Phase 2)
│
├── models/        ← Save trained ML .pkl files here (Phase 4)
├── tests/
│   └── test_engines.py
│
├── .env.example   ← Copy to .env, add your API keys
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure LLM (choose one)
cp .env.example .env
# Option A: add GROQ_API_KEY in .env
# Option B: use local Ollama (no key), then set OLLAMA_MODEL in .env

# 4. Run the backend
uvicorn backend.main:app --reload
# API docs: http://localhost:8000/docs

# 5. Open frontend
open frontend/index.html        # or double-click in file manager
```

---

## Phase 2 — RAG Knowledge Engine

```bash
# Install dependencies
pip install chromadb sentence-transformers

# (Optional) Add your own knowledge files to data/raw/
# Good sources:
#   OWASP Cheat Sheet Series : https://cheatsheetseries.owasp.org/
#   NVD CVE data             : https://nvd.nist.gov/vuln/data-feeds
#   MITRE ATT&CK             : https://attack.mitre.org/

# Seed the knowledge base (built-in knowledge included automatically)
python scripts/seed_knowledge.py

# Re-index after adding new files
python scripts/seed_knowledge.py --force

# Restart the server — RAG is now active
uvicorn backend.main:app --reload
```

Phase 2 activates automatically when `chromadb` and `sentence-transformers`
are installed. If they are missing, the engine falls back to pure LLM answering.

---

## Phase 3 — IOC Enrichment (Real Threat Intelligence)

```bash
# Install dependency
pip install requests

# Add API keys to your .env file:
#   VIRUSTOTAL_API_KEY  — https://www.virustotal.com/gui/join-us  (500 req/day free)
#   ABUSEIPDB_API_KEY   — https://www.abuseipdb.com/register      (1000 req/day free)
#   SHODAN_API_KEY      — https://account.shodan.io/register       (100 req/month free)

# Restart server — IOC enrichment is now active for phishing + log queries
uvicorn backend.main:app --reload
```

Phase 3 activates automatically when `requests` is installed and at least
one API key is present. Missing keys = that provider is silently skipped.

---

## Test the API directly

```bash
# Phishing check (triggers IOC enrichment in Phase 3)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Check this: http://paypal-verify.xyz/login urgent account issue"}'

# Log analysis
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Failed password for root from 185.220.101.5 — 47 times in 90s"}'

# Knowledge query (uses RAG in Phase 2)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How does SQL injection work and how do I prevent it?"}'
```

---

## Run Tests

```bash
pip install pytest
python -m pytest tests/ -v
# All tests pass without any API keys
```

---

## Resume / LinkedIn Description

```
SentinelForge — Cybersecurity Threat Intelligence Copilot
Python · FastAPI · ChromaDB · sentence-transformers · Anthropic Claude API

• Multi-engine architecture: NLP intent routing → specialised detection engines → LLM explanation
• Phishing engine: rule-based URL/content feature extraction with confidence scoring
• Log analysis engine: detects brute force, credential stuffing, privilege escalation with MITRE ATT&CK mapping
• RAG pipeline: semantic search over OWASP/CVE knowledge base (ChromaDB + all-MiniLM-L6-v2)
• IOC enrichment: real-time threat intelligence via VirusTotal, AbuseIPDB, and Shodan APIs
• LLM layer: Claude generates structured, human-readable threat explanations with remediation steps
• REST API (FastAPI) + professional dark-mode chat frontend (vanilla HTML/CSS/JS)
```
