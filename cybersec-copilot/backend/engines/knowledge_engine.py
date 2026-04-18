"""
RAG Knowledge Engine  —  Phase 2
==================================
Retrieval-Augmented Generation using ChromaDB + sentence-transformers.

How it works:
  1. Drop .txt knowledge files into data/raw/
  2. Run: python scripts/seed_knowledge.py
     → Splits text into chunks, embeds them, stores in ChromaDB (local on disk)
  3. At query time: embed the question → find nearest chunks → pass to LLM

Install dependencies:
  pip install chromadb sentence-transformers
"""

from __future__ import annotations
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[3]
DATA_RAW   = ROOT / "data" / "raw"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "cybersec_knowledge"

_collection  = None
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("Run: pip install sentence-transformers")
        print("[RAG] Loading embedding model (~80 MB on first run)...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb
    except ImportError:
        raise RuntimeError("Run: pip install chromadb")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _chunk_text(text: str, max_words: int = 120, overlap: int = 20) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)
        i += max_words - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 40]


def _builtin_chunks() -> list[dict]:
    texts = [
        "OWASP Top 10 lists the ten most critical web security risks: A01 Broken Access Control, A02 Cryptographic Failures, A03 Injection, A04 Insecure Design, A05 Security Misconfiguration, A06 Vulnerable Components, A07 Auth Failures, A08 Integrity Failures, A09 Logging Failures, A10 SSRF.",
        "SQL Injection (SQLi) OWASP A03 — attacker inserts malicious SQL into input fields. Can read, modify, or delete data. Prevention: parameterised queries, prepared statements, input validation, least-privilege DB accounts.",
        "Cross-Site Scripting (XSS) OWASP A03 — malicious scripts injected into trusted websites, executed in victim browsers. Types: Stored, Reflected, DOM-based. Prevention: output encoding, Content Security Policy, HttpOnly cookies.",
        "CSRF (Cross-Site Request Forgery) — forces authenticated users to submit unwanted requests. Prevention: CSRF tokens, SameSite cookie attribute (Strict/Lax), double-submit cookie.",
        "Broken Access Control OWASP A01 — most common risk. Users act outside intended permissions. Examples: IDOR, privilege escalation. Prevention: deny by default, server-side checks, log failures.",
        "Phishing — social engineering via email/SMS/voice impersonating trusted entities to steal credentials or install malware. Indicators: urgency, mismatched URLs, generic greetings. Prevention: MFA, DMARC, user training.",
        "Brute Force MITRE T1110 — trying all password combinations. Variants: dictionary, credential stuffing, password spraying. Defence: account lockout, rate limiting, MFA, fail2ban.",
        "Man-in-the-Middle (MITM) — intercepts traffic between two parties. Methods: ARP spoofing, SSL stripping, rogue Wi-Fi. Prevention: TLS/HTTPS, HSTS, certificate pinning, VPN.",
        "Ransomware — encrypts files and demands payment for decryption key. Delivered via phishing or RDP. Prevention: offline backups, network segmentation, patch management, EDR. MITRE T1486.",
        "Zero-day — unknown vulnerability with no patch. Detected via behaviour-based EDR and anomaly detection rather than signatures.",
        "MITRE ATT&CK — knowledge base of adversary tactics and techniques from real observations. Organised as Tactics (why) > Techniques (how) > Sub-techniques. Used for threat modelling and detection engineering.",
        "CVE (Common Vulnerabilities and Exposures) — unique IDs for known security flaws, format CVE-YEAR-NUMBER. Scored by CVSS 0-10. Critical = 9.0-10.0. Search at nvd.nist.gov.",
        "Defence-in-Depth — layered security: perimeter (firewall), network (IDS/IPS), host (EDR), application (WAF), data (encryption), people (training).",
        "Incident Response phases: Preparation, Identification, Containment, Eradication, Recovery, Lessons Learned. Tools: SIEM, EDR, SOAR, threat intelligence platforms.",
        "Log analysis: key sources are auth.log (SSH/login), syslog (system), web access logs, Windows Event Logs (4625=failed login). Brute force: many failures from one IP. Lateral movement: logins across many internal hosts.",
        "Network reconnaissance MITRE T1046/T1595 — tools: nmap, masscan. Indicators: sequential port probing, SYN without handshake completion. Detection: firewall logs, IDS signatures.",
    ]
    return [{"text": t, "source": "builtin"} for t in texts]


def seed(force: bool = False) -> int:
    collection = _get_collection()
    if collection.count() > 0 and not force:
        print(f"[RAG] Already seeded ({collection.count()} chunks). Use force=True to re-index.")
        return 0

    model = _get_embed_model()
    txt_files = sorted(DATA_RAW.glob("*.txt"))
    all_chunks = _builtin_chunks()

    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        for chunk in _chunk_text(text):
            all_chunks.append({"text": chunk, "source": txt_file.name})

    batch_size = 64
    added = 0
    for start in range(0, len(all_chunks), batch_size):
        batch  = all_chunks[start : start + batch_size]
        texts  = [c["text"] for c in batch]
        embeds = model.encode(texts, show_progress_bar=False).tolist()
        ids    = [f"chunk_{start + i}" for i in range(len(batch))]
        metas  = [{"source": c["source"]} for c in batch]
        collection.add(documents=texts, embeddings=embeds, ids=ids, metadatas=metas)
        added += len(batch)

    print(f"[RAG] Seeded {added} chunks ({len(txt_files)} file(s) + built-in).")
    return added


def query(question: str, top_k: int = 4) -> list[str]:
    collection = _get_collection()
    if collection.count() == 0:
        seed()
    model  = _get_embed_model()
    q_vec  = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=min(top_k, collection.count()),
        include=["documents"],
    )
    return results["documents"][0] if results["documents"] else []


def collection_size() -> int:
    return _get_collection().count()
