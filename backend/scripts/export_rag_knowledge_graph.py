"""Backward-compatible entry point for the canonical graph corpus builder."""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "scripts" / "build_knowledge_graph.py"), run_name="__main__")
