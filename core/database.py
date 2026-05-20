import sqlite3
import threading
import numpy as np
from pathlib import Path

DB_PATH = Path("data/quranid.db")
_local = threading.local()


def _get_conn():
    if not hasattr(_local, 'conn') or _local.conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _local.conn = sqlite3.connect(str(DB_PATH))
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA foreign_keys = ON")
    return _local.conn


def init_db() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reciters (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT UNIQUE NOT NULL,
            display_name TEXT DEFAULT '',
            arabic_name  TEXT DEFAULT '',
            style        TEXT DEFAULT '',
            nationality  TEXT DEFAULT '',
            enrolled_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            reciter_id  INTEGER NOT NULL REFERENCES reciters(id) ON DELETE CASCADE,
            embedding   BLOB NOT NULL,
            source_file TEXT DEFAULT '',
            created_at  TEXT DEFAULT (datetime('now'))
        );

    """)
    # migration: add display_name to existing databases that predate this column
    try:
        conn.execute("ALTER TABLE reciters ADD COLUMN display_name TEXT DEFAULT ''")
        conn.commit()
    except Exception:
        pass  # column already exists

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            query_file    TEXT,
            matched_name  TEXT,
            score         REAL,
            method        TEXT,
            identified_at TEXT DEFAULT (datetime('now'))
        );
    """)


def add_reciter(name: str, style: str = "", nationality: str = "", arabic_name: str = "",
                display_name: str = "") -> int:
    conn = _get_conn()
    dn = display_name or name
    conn.execute(
        "INSERT OR IGNORE INTO reciters (name, display_name, arabic_name, style, nationality) "
        "VALUES (?, ?, ?, ?, ?)",
        (name, dn, arabic_name, style, nationality),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM reciters WHERE name = ?", (name,)).fetchone()
    return row["id"]


def save_embedding(reciter_id: int, embedding: np.ndarray, source_file: str = "") -> None:
    conn = _get_conn()
    blob = embedding.astype(np.float32).tobytes()
    conn.execute(
        "INSERT INTO embeddings (reciter_id, embedding, source_file) VALUES (?, ?, ?)",
        (reciter_id, blob, source_file),
    )
    conn.commit()


def get_all_embeddings() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("""
        SELECT e.reciter_id, r.name, r.display_name, e.embedding
        FROM embeddings e
        JOIN reciters r ON r.id = e.reciter_id
    """).fetchall()
    return [
        {
            "reciter_id":   row["reciter_id"],
            "name":         row["name"],
            "display_name": row["display_name"] or row["name"],
            "embedding":    np.frombuffer(row["embedding"], dtype=np.float32).copy(),
        }
        for row in rows
    ]


def list_reciters() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("""
        SELECT r.name, r.arabic_name, r.style, r.nationality, r.enrolled_at,
               COUNT(e.id) AS sample_count
        FROM reciters r
        LEFT JOIN embeddings e ON e.reciter_id = r.id
        GROUP BY r.id
        ORDER BY r.name
    """).fetchall()
    return [dict(row) for row in rows]


def update_reciter(name: str, arabic_name: str) -> bool:
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE reciters SET arabic_name = ? WHERE name = ?",
        (arabic_name, name),
    )
    conn.commit()
    return cur.rowcount > 0


def remove_reciter(name: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM reciters WHERE name = ?", (name,))
    conn.commit()
    return cur.rowcount > 0


def save_history(query_file: str, matched_name: str, score: float, method: str) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO history (query_file, matched_name, score, method) VALUES (?, ?, ?, ?)",
        (query_file, matched_name, score, method),
    )
    conn.commit()


def get_history(limit: int = 20) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT query_file, matched_name, score, method, identified_at "
        "FROM history ORDER BY identified_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]
