"""
Analytics module — SQLite-based event tracking for the voice agent.
Tracks: sessions, queries, property searches, errors, barge-ins, voice changes, etc.
"""

import os
import json
import time
import uuid
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Optional

DB_PATH = os.getenv("ANALYTICS_DB", os.path.join(os.path.dirname(__file__), "..", "analytics.db"))

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
    return _local.conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            ip_address TEXT,
            user_agent TEXT,
            voice TEXT DEFAULT 'test.wav',
            mode TEXT DEFAULT 'voice',
            total_queries INTEGER DEFAULT 0,
            total_properties_shown INTEGER DEFAULT 0,
            total_barge_ins INTEGER DEFAULT 0,
            duration_seconds REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            data TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            query_text TEXT NOT NULL,
            query_mode TEXT DEFAULT 'voice',
            is_property_query INTEGER DEFAULT 0,
            properties_returned INTEGER DEFAULT 0,
            llm_response TEXT,
            response_time_ms INTEGER DEFAULT 0,
            error TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT NOT NULL,
            error_type TEXT NOT NULL,
            error_message TEXT,
            context TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
        CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_queries_session ON queries(session_id);
        CREATE INDEX IF NOT EXISTS idx_queries_ts ON queries(timestamp);
        CREATE INDEX IF NOT EXISTS idx_errors_ts ON errors(timestamp);
    """)
    conn.commit()
    print("[ANALYTICS] Database initialized")


# ── Session Management ──

def create_session(session_id: str, ip: str = "", user_agent: str = "", voice: str = "test.wav") -> str:
    conn = _get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO sessions (id, started_at, ip_address, user_agent, voice) VALUES (?, ?, ?, ?, ?)",
        (session_id, datetime.utcnow().isoformat(), ip, user_agent, voice)
    )
    conn.commit()
    return session_id


def end_session(session_id: str):
    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
        (now, session_id)
    )
    # Calculate duration
    row = conn.execute("SELECT started_at, ended_at FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row and row["started_at"] and row["ended_at"]:
        try:
            start = datetime.fromisoformat(row["started_at"])
            end = datetime.fromisoformat(row["ended_at"])
            dur = (end - start).total_seconds()
            conn.execute("UPDATE sessions SET duration_seconds = ? WHERE id = ?", (dur, session_id))
        except Exception:
            pass
    conn.commit()


def update_session_voice(session_id: str, voice: str):
    conn = _get_conn()
    conn.execute("UPDATE sessions SET voice = ? WHERE id = ?", (voice, session_id))
    conn.commit()


# ── Event Logging ──

def log_event(session_id: str, event_type: str, data: dict = None):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO events (session_id, event_type, timestamp, data) VALUES (?, ?, ?, ?)",
        (session_id, event_type, datetime.utcnow().isoformat(), json.dumps(data) if data else None)
    )
    conn.commit()


def log_query(session_id: str, query_text: str, mode: str = "voice",
              is_property_query: bool = False, properties_returned: int = 0,
              llm_response: str = "", response_time_ms: int = 0, error: str = ""):
    conn = _get_conn()
    conn.execute(
        """INSERT INTO queries (session_id, timestamp, query_text, query_mode,
           is_property_query, properties_returned, llm_response, response_time_ms, error)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, datetime.utcnow().isoformat(), query_text, mode,
         1 if is_property_query else 0, properties_returned,
         llm_response[:500] if llm_response else "", response_time_ms,
         error or None)
    )
    # Update session counters
    conn.execute("UPDATE sessions SET total_queries = total_queries + 1 WHERE id = ?", (session_id,))
    if properties_returned > 0:
        conn.execute(
            "UPDATE sessions SET total_properties_shown = total_properties_shown + ? WHERE id = ?",
            (properties_returned, session_id)
        )
    conn.commit()


def log_barge_in(session_id: str):
    conn = _get_conn()
    conn.execute("UPDATE sessions SET total_barge_ins = total_barge_ins + 1 WHERE id = ?", (session_id,))
    conn.commit()
    log_event(session_id, "barge_in")


def log_error(session_id: str, error_type: str, error_message: str, context: str = ""):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO errors (session_id, timestamp, error_type, error_message, context) VALUES (?, ?, ?, ?, ?)",
        (session_id or "", datetime.utcnow().isoformat(), error_type, error_message, context or None)
    )
    conn.commit()


# ── Analytics Queries ──

def get_dashboard_stats(days: int = 7) -> dict:
    conn = _get_conn()
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    total_sessions = conn.execute(
        "SELECT COUNT(*) as c FROM sessions WHERE started_at >= ?", (since,)
    ).fetchone()["c"]

    total_queries = conn.execute(
        "SELECT COUNT(*) as c FROM queries WHERE timestamp >= ?", (since,)
    ).fetchone()["c"]

    total_property_queries = conn.execute(
        "SELECT COUNT(*) as c FROM queries WHERE timestamp >= ? AND is_property_query = 1", (since,)
    ).fetchone()["c"]

    total_errors = conn.execute(
        "SELECT COUNT(*) as c FROM errors WHERE timestamp >= ?", (since,)
    ).fetchone()["c"]

    failed_queries = conn.execute(
        "SELECT COUNT(*) as c FROM queries WHERE timestamp >= ? AND error IS NOT NULL AND error != ''", (since,)
    ).fetchone()["c"]

    avg_response_time = conn.execute(
        "SELECT AVG(response_time_ms) as avg FROM queries WHERE timestamp >= ? AND response_time_ms > 0", (since,)
    ).fetchone()["avg"] or 0

    total_barge_ins = conn.execute(
        "SELECT SUM(total_barge_ins) as c FROM sessions WHERE started_at >= ?", (since,)
    ).fetchone()["c"] or 0

    avg_duration = conn.execute(
        "SELECT AVG(duration_seconds) as avg FROM sessions WHERE started_at >= ? AND duration_seconds > 0", (since,)
    ).fetchone()["avg"] or 0

    # Daily breakdown
    daily = conn.execute("""
        SELECT DATE(started_at) as day, COUNT(*) as sessions,
               (SELECT COUNT(*) FROM queries q WHERE DATE(q.timestamp) = DATE(s.started_at)) as queries
        FROM sessions s WHERE started_at >= ?
        GROUP BY DATE(started_at) ORDER BY day DESC LIMIT 14
    """, (since,)).fetchall()

    # Top queries
    top_queries = conn.execute("""
        SELECT query_text, COUNT(*) as cnt, AVG(response_time_ms) as avg_time
        FROM queries WHERE timestamp >= ?
        GROUP BY query_text ORDER BY cnt DESC LIMIT 20
    """, (since,)).fetchall()

    # Voice usage
    voice_usage = conn.execute("""
        SELECT voice, COUNT(*) as cnt FROM sessions
        WHERE started_at >= ? GROUP BY voice ORDER BY cnt DESC
    """, (since,)).fetchall()

    return {
        "period_days": days,
        "total_sessions": total_sessions,
        "total_queries": total_queries,
        "total_property_queries": total_property_queries,
        "total_errors": total_errors,
        "failed_queries": failed_queries,
        "avg_response_time_ms": round(avg_response_time),
        "total_barge_ins": total_barge_ins,
        "avg_session_duration_sec": round(avg_duration),
        "daily": [dict(r) for r in daily],
        "top_queries": [dict(r) for r in top_queries],
        "voice_usage": [dict(r) for r in voice_usage],
    }


def get_recent_sessions(limit: int = 50, offset: int = 0) -> list:
    conn = _get_conn()
    rows = conn.execute("""
        SELECT * FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?
    """, (limit, offset)).fetchall()
    return [dict(r) for r in rows]


def get_session_detail(session_id: str) -> dict:
    conn = _get_conn()
    session = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not session:
        return {}
    events = conn.execute(
        "SELECT * FROM events WHERE session_id = ? ORDER BY timestamp", (session_id,)
    ).fetchall()
    queries = conn.execute(
        "SELECT * FROM queries WHERE session_id = ? ORDER BY timestamp", (session_id,)
    ).fetchall()
    errors = conn.execute(
        "SELECT * FROM errors WHERE session_id = ? ORDER BY timestamp", (session_id,)
    ).fetchall()
    return {
        "session": dict(session),
        "events": [dict(r) for r in events],
        "queries": [dict(r) for r in queries],
        "errors": [dict(r) for r in errors],
    }


def get_recent_queries(limit: int = 100, offset: int = 0, failed_only: bool = False) -> list:
    conn = _get_conn()
    if failed_only:
        rows = conn.execute("""
            SELECT q.*, s.ip_address, s.voice FROM queries q
            LEFT JOIN sessions s ON q.session_id = s.id
            WHERE q.error IS NOT NULL AND q.error != ''
            ORDER BY q.timestamp DESC LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()
    else:
        rows = conn.execute("""
            SELECT q.*, s.ip_address, s.voice FROM queries q
            LEFT JOIN sessions s ON q.session_id = s.id
            ORDER BY q.timestamp DESC LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()
    return [dict(r) for r in rows]


def get_recent_errors(limit: int = 50) -> list:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM errors ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]
