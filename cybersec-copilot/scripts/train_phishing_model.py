"""
Phase 4 — Train the ML Phishing Classifier
============================================
Trains a RandomForest model on URL + content features.
Saves model to models/phishing_classifier.pkl

Run:
  python scripts/train_phishing_model.py

If you have a real dataset (CSV), place it at:
  data/raw/phishing_dataset.csv
  Required columns: url, label  (label: 1=phishing, 0=clean)

Free datasets to download:
  - PhiUSIIL Phishing URL Dataset : https://www.kaggle.com/datasets/harisudhan411/phishing-and-legitimate-urls
  - UCI Phishing Websites         : https://archive.ics.uci.edu/dataset/327/phishing+websites
  - Mendeley URL dataset          : https://data.mendeley.com/datasets/c2gd2jdhnz

Without a dataset this script generates synthetic training data so you
can see the full pipeline working immediately.
"""

import sys
import re
import math
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

ROOT       = Path(__file__).resolve().parents[1]
DATA_PATH  = ROOT / "data" / "raw" / "phishing_dataset.csv"
MODEL_DIR  = ROOT / "models"
MODEL_PATH = MODEL_DIR / "phishing_classifier.pkl"
META_PATH  = MODEL_DIR / "phishing_classifier_meta.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Feature extraction ────────────────────────────────────────────────────────

SUSPICIOUS_WORDS = [
    "verify", "account", "secure", "update", "login", "banking", "confirm",
    "password", "credit", "urgent", "suspended", "click", "limited", "free",
    "winner", "congratulations", "prize", "offer", "alert", "warning",
]

TRUSTED_TLDS = {".com", ".org", ".net", ".gov", ".edu", ".co.uk", ".io"}

BRAND_NAMES = [
    "paypal", "amazon", "google", "apple", "microsoft", "netflix", "facebook",
    "instagram", "twitter", "bank", "chase", "wellsfargo", "citibank",
    "linkedin", "dropbox", "adobe", "yahoo", "outlook", "office365",
]


def extract_url_features(url: str) -> dict:
    """
    Extract 20 numerical features from a URL.
    All features are things a real phishing classifier uses.
    """
    url_lower = url.lower()

    # Parse parts
    has_https    = int(url_lower.startswith("https://"))
    has_http_only= int(url_lower.startswith("http://") and not has_https)

    try:
        domain_part = re.sub(r"https?://", "", url_lower).split("/")[0]
        path_part   = "/" + "/".join(re.sub(r"https?://", "", url_lower).split("/")[1:])
    except Exception:
        domain_part = url_lower
        path_part   = ""

    subdomain_count = max(0, domain_part.count(".") - 1)
    domain_len      = len(domain_part)
    url_len         = len(url)
    path_len        = len(path_part)

    has_ip = int(bool(re.match(r"\d{1,3}(\.\d{1,3}){3}", domain_part)))

    # Suspicious characters
    at_count     = url.count("@")
    dash_count   = domain_part.count("-")
    dot_count    = url.count(".")
    digit_count  = sum(c.isdigit() for c in domain_part)
    special_count= sum(1 for c in url if c in "%_=?&#")

    # Encoded chars (obfuscation)
    hex_encoded = len(re.findall(r"%[0-9a-fA-F]{2}", url))

    # Brand name in suspicious position
    brand_in_domain = int(any(b in domain_part for b in BRAND_NAMES))
    brand_in_path   = int(any(b in path_part   for b in BRAND_NAMES) and not brand_in_domain)

    # TLD suspicion
    tld = "." + domain_part.split(".")[-1] if "." in domain_part else ""
    suspicious_tld = int(tld in {".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".click"})
    trusted_tld    = int(tld in TRUSTED_TLDS)

    # Entropy of domain (high entropy = random-looking = suspicious)
    def entropy(s):
        if not s: return 0
        freq = {c: s.count(c)/len(s) for c in set(s)}
        return -sum(p * math.log2(p) for p in freq.values())

    domain_entropy = entropy(domain_part.replace(".", ""))

    # Subdomain depth
    has_deep_subdomain = int(subdomain_count >= 3)

    return {
        "url_length":         url_len,
        "domain_length":      domain_len,
        "path_length":        path_len,
        "has_https":          has_https,
        "has_http_only":      has_http_only,
        "has_ip_address":     has_ip,
        "subdomain_count":    subdomain_count,
        "has_deep_subdomain": has_deep_subdomain,
        "at_sign_count":      at_count,
        "dash_in_domain":     dash_count,
        "dot_count":          dot_count,
        "digit_in_domain":    digit_count,
        "special_char_count": special_count,
        "hex_encoded_chars":  hex_encoded,
        "brand_in_domain":    brand_in_domain,
        "brand_in_path":      brand_in_path,
        "suspicious_tld":     suspicious_tld,
        "trusted_tld":        trusted_tld,
        "domain_entropy":     round(domain_entropy, 4),
        "suspicious_words_in_url": sum(w in url_lower for w in SUSPICIOUS_WORDS),
    }


FEATURE_NAMES = list(extract_url_features("https://example.com").keys())


# ── Synthetic dataset generator (used when no real CSV exists) ────────────────

def _make_synthetic_dataset(n_phishing=1800, n_clean=1800) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset for demonstration.
    Real model: replace with a downloaded CSV of real phishing URLs.
    """
    random.seed(42)
    rows = []

    # Phishing URL patterns
    phishing_templates = [
        "http://{ip}/paypal/login.php",
        "http://{brand}-secure-{rand}.{stld}/verify",
        "http://{rand}.{stld}/account/suspended?id={rand}",
        "https://{brand}.{rand}-verify.com/login",
        "http://{rand}.tk/win-prize/claim?user={rand}",
        "http://{ip}:{port}/microsoft/signin",
        "https://secure-{brand}-{rand}.xyz/update",
        "http://{rand}-{rand}-{rand}.ml/banking/confirm",
    ]

    clean_templates = [
        "https://www.{brand}.com",
        "https://{brand}.com/products/{rand}",
        "https://blog.{brand}.io/article/{rand}",
        "https://api.{brand}.net/v2/data",
        "https://www.{brand}.org/about",
        "https://docs.{brand}.com/guide/{rand}",
        "https://support.{brand}.co.uk/help/{rand}",
    ]

    clean_brands = ["github", "stackoverflow", "wikipedia", "mozilla", "python",
                    "numpy", "django", "fastapi", "cloudflare", "ubuntu"]

    phish_brands = random.choices(BRAND_NAMES, k=n_phishing)
    stlds = ["tk", "ml", "ga", "cf", "xyz", "top", "click"]

    def rstr(k=6): return "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=k))
    def rip(): return f"{random.randint(1,254)}.{random.randint(0,254)}.{random.randint(0,254)}.{random.randint(1,254)}"

    for i in range(n_phishing):
        tmpl  = random.choice(phishing_templates)
        brand = phish_brands[i]
        url   = tmpl.format(brand=brand, rand=rstr(), ip=rip(),
                            stld=random.choice(stlds), port=random.randint(8000,9999))
        feats = extract_url_features(url)
        feats["label"] = 1
        rows.append(feats)

    for _ in range(n_clean):
        tmpl  = random.choice(clean_templates)
        brand = random.choice(clean_brands)
        url   = tmpl.format(brand=brand, rand=rstr())
        feats = extract_url_features(url)
        feats["label"] = 0
        rows.append(feats)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ── Training ──────────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        print(f"[Train] Loading real dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)

        # Normalise column names
        df.columns = [c.lower().strip() for c in df.columns]

        if "url" in df.columns and "label" in df.columns:
            print(f"[Train] Extracting features from {len(df)} URLs…")
            feature_rows = [extract_url_features(str(u)) for u in df["url"]]
            feat_df = pd.DataFrame(feature_rows)
            feat_df["label"] = df["label"].values
            return feat_df
        else:
            print(f"[Train] WARNING: CSV must have 'url' and 'label' columns. "
                  "Falling back to synthetic data.")

    print("[Train] No dataset found — generating synthetic data.")
    print("[Train] For a real model download a dataset from Kaggle (see script header).")
    return _make_synthetic_dataset()


def train():
    print("=" * 56)
    print("  SentinelForge — Phase 4 ML Phishing Classifier")
    print("=" * 56)

    df = load_dataset()
    print(f"\n[Train] Dataset: {len(df)} samples  |  "
          f"Phishing: {df['label'].sum()}  |  Clean: {(df['label']==0).sum()}")

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Two models — RF is faster, GBM is slightly more accurate
    print("\n[Train] Training RandomForest…")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"[Train] CV AUC-ROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Test set evaluation
    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n[Train] Test AUC-ROC : {auc:.4f}")
    print(f"[Train] Test Accuracy: {(y_pred == y_test).mean():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Clean", "Phishing"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n  True Neg: {cm[0,0]}  False Pos: {cm[0,1]}")
    print(f"  False Neg: {cm[1,0]}  True Pos: {cm[1,1]}")

    # Feature importance
    importances = sorted(
        zip(FEATURE_NAMES, rf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\nTop 10 most important features:")
    for name, imp in importances[:10]:
        bar = "█" * int(imp * 60)
        print(f"  {name:<28} {bar}  {imp:.4f}")

    # Save model
    joblib.dump(rf, MODEL_PATH)
    meta = {
        "feature_names": FEATURE_NAMES,
        "auc_roc": round(auc, 4),
        "accuracy": round((y_pred == y_test).mean(), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "using_synthetic": not DATA_PATH.exists(),
        "top_features": [name for name, _ in importances[:5]],
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"\n[Train] Model saved to {MODEL_PATH}")
    print(f"[Train] Metadata saved to {META_PATH}")
    print("\n✓ Training complete. Restart the API server to activate ML detection.")

    return rf, meta


if __name__ == "__main__":
    train()
