from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "data" / "chat_history.sqlite3"
_LOCK = Lock()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema() -> None:
    with _LOCK:
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    audit_id TEXT,
                    intent TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    response_json TEXT NOT NULL
                )
                """
            )
            conn.commit()


def _dump_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    if isinstance(response, dict):
        return response
    raise TypeError("response must be a dict-like object or Pydantic model")


def append_chat_exchange(message: str, response: Any) -> None:
    _ensure_schema()
    response_data = _dump_response(response)
    with _LOCK:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_history (audit_id, intent, user_message, response_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    response_data.get("audit_id"),
                    response_data.get("intent", "unknown"),
                    message,
                    json.dumps(response_data, ensure_ascii=True),
                ),
            )
            conn.commit()


def fetch_chat_history(limit: int = 50) -> list[dict[str, Any]]:
    _ensure_schema()
    safe_limit = max(1, min(int(limit), 200))

    with _LOCK:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, audit_id, intent, user_message, response_json
                FROM chat_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

    items: list[dict[str, Any]] = []
    for row in reversed(rows):
        response_data = json.loads(row["response_json"])
        items.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "audit_id": row["audit_id"],
                "intent": row["intent"],
                "message": row["user_message"],
                "response": response_data,
            }
        )

    return items