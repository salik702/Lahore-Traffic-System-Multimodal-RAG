"""Persist digital-twin experiments separately from real-world traffic records."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class ExperimentStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS twin_experiments (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                controller TEXT NOT NULL,
                scenario TEXT NOT NULL,
                seed INTEGER NOT NULL,
                demand_scale REAL NOT NULL,
                duration REAL NOT NULL,
                engine TEXT NOT NULL,
                config TEXT NOT NULL,
                metrics TEXT NOT NULL,
                comparison TEXT,
                notes TEXT,
                created_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS twin_training (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                episodes INTEGER,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
                )"""
            )

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def save_experiment(self, kind: str, controller: str, scenario: str, seed: int, demand_scale: float, duration: float, engine: str, config: dict, metrics: dict, comparison: dict | None = None, notes: str = "") -> str:
        experiment_id = uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO twin_experiments(id, kind, controller, scenario, seed, demand_scale, duration, engine, config, metrics, comparison, notes, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (experiment_id, kind, controller, scenario, seed, demand_scale, duration, engine, json.dumps(config), json.dumps(metrics), json.dumps(comparison) if comparison else None, notes, datetime.now(timezone.utc).isoformat()),
            )
        return experiment_id

    def list_experiments(self, limit: int = 40) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, kind, controller, scenario, seed, demand_scale, duration, engine, metrics, comparison, notes, created_at FROM twin_experiments ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        items = []
        for row in rows:
            items.append({
                "id": row[0], "kind": row[1], "controller": row[2], "scenario": row[3],
                "seed": row[4], "demand_scale": row[5], "duration": row[6], "engine": row[7],
                "metrics": json.loads(row[8]), "comparison": json.loads(row[9]) if row[9] else None,
                "notes": row[10], "created_at": row[11], "provenance": "simulation",
            })
        return items

    def get(self, experiment_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT id, kind, controller, scenario, seed, demand_scale, duration, engine, config, metrics, comparison, notes, created_at FROM twin_experiments WHERE id=?", (experiment_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "kind": row[1], "controller": row[2], "scenario": row[3],
            "seed": row[4], "demand_scale": row[5], "duration": row[6], "engine": row[7],
            "config": json.loads(row[8]), "metrics": json.loads(row[9]),
            "comparison": json.loads(row[10]) if row[10] else None, "notes": row[11],
            "created_at": row[12], "provenance": "simulation",
        }

    def save_training(self, status: str, payload: dict, training_id: str | None = None, episodes: int | None = None) -> str:
        training_id = training_id or uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            existing = conn.execute("SELECT id FROM twin_training WHERE id=?", (training_id,)).fetchone()
            if existing:
                conn.execute("UPDATE twin_training SET status=?, episodes=?, payload=?, updated_at=? WHERE id=?", (status, episodes, json.dumps(payload), now, training_id))
            else:
                conn.execute("INSERT INTO twin_training(id, status, episodes, payload, created_at, updated_at) VALUES (?,?,?,?,?,?)", (training_id, status, episodes, json.dumps(payload), now, now))
        return training_id

    def latest_training(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT id, status, episodes, payload, created_at, updated_at FROM twin_training ORDER BY updated_at DESC LIMIT 1").fetchone()
        if not row:
            return None
        return {"id": row[0], "status": row[1], "episodes": row[2], **json.loads(row[3]), "created_at": row[4], "updated_at": row[5]}
