import os
import sqlite3
import json
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from contextlib import contextmanager

from app.config import settings


def resolve_db_path(db_url: str) -> str:
    """
    Extracts the filesystem path from a sqlite URL and ensures directory exists.
    Supports sqlite:////data/app.db (Unix absolute)
    and sqlite:///./data/app.db (relative)
    """
    if db_url.startswith("sqlite:////"):
        path = "/" + db_url[len("sqlite:////"):]
    elif db_url.startswith("sqlite:///"):
        path = db_url[len("sqlite:///"):]
    else:
        path = db_url

    # Check if directory can be created or if path is on root /data in Windows
    abs_path = os.path.abspath(path)
    dir_path = os.path.dirname(abs_path)

    try:
        os.makedirs(dir_path, exist_ok=True)
        # Test file creation ability
        test_file = os.path.join(dir_path, ".write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        return abs_path
    except Exception:
        # Fallback to local ./data/app.db if root path /data fails (e.g. Windows non-admin)
        local_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(local_dir, exist_ok=True)
        return os.path.join(local_dir, "app.db")


DB_PATH = resolve_db_path(settings.DATABASE_URL)


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=20.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """
    Initializes the SQLite database schema automatically on startup.
    Matches the assignment contract exactly.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        # Minimal data model required by specification
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            );
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id);
        """)
        # Session table to persist session-specific system prompt
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                system_prompt TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)


def is_db_reachable() -> bool:
    """Verifies SQLite is accessible and can execute queries."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            row = cursor.fetchone()
            return row is not None and row[0] == 1
    except Exception:
        return False


def get_session_system_prompt(session_id: str) -> Optional[str]:
    """Retrieves session-specific system prompt if set."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT system_prompt FROM sessions WHERE session_id = ?;", (session_id,))
        row = cursor.fetchone()
        if row and row["system_prompt"]:
            return row["system_prompt"]
        return None


def save_session_system_prompt(session_id: str, system_prompt: str):
    """Saves or updates system prompt for a session."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (session_id, system_prompt, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                system_prompt = excluded.system_prompt,
                updated_at = excluded.updated_at;
        """, (session_id, system_prompt, now_iso, now_iso))


def session_exists(session_id: str) -> bool:
    """Returns True if the session has any messages stored."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM messages WHERE session_id = ? LIMIT 1;", (session_id,))
        return cursor.fetchone() is not None


def get_session_messages(session_id: str) -> List[dict]:
    """
    Returns messages oldest first for session_id.
    Excludes system prompts.
    Parses assistant metadata JSON string into dict.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, session_id, role, content, metadata, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC;
        """, (session_id,))
        rows = cursor.fetchall()
        result = []
        for r in rows:
            meta = None
            if r["metadata"]:
                try:
                    meta = json.loads(r["metadata"])
                except Exception:
                    meta = None
            result.append({
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "metadata": meta,
                "created_at": r["created_at"]
            })
        return result


def save_message(session_id: str, role: str, content: str, metadata: Optional[dict] = None) -> int:
    """
    Inserts a new message into messages table.
    created_at is ISO-8601 server timestamp.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    meta_json = json.dumps(metadata) if metadata is not None else None
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO messages (session_id, role, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?);
        """, (session_id, role, content, meta_json, now_iso))
        return cursor.lastrowid
