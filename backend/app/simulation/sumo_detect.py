"""Detect Eclipse SUMO binaries without pretending they are installed."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from app.core.config import Settings


def candidate_homes(settings: Settings) -> list[Path]:
    homes: list[Path] = []
    env = os.environ.get("SUMO_HOME", "").strip()
    if env:
        homes.append(Path(env))
    configured = getattr(settings, "sumo_home", "") or ""
    if configured:
        homes.append(Path(configured))
    homes.append(settings.project_root / "tools" / "sumo")
    homes.append(Path("C:/Program Files/Eclipse SUMO"))
    homes.append(Path("C:/Program Files (x86)/Eclipse SUMO"))
    unique: list[Path] = []
    seen: set[str] = set()
    for home in homes:
        key = str(home.resolve()) if home.exists() else str(home)
        if key not in seen:
            seen.add(key)
            unique.append(home)
    return unique


def resolve_bin(name: str, home: Path | None) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    if home:
        for folder in (home / "bin", home):
            for candidate in (folder / f"{name}.exe", folder / name):
                if candidate.is_file():
                    return str(candidate)
    return None


def detect_sumo(settings: Settings) -> dict:
    for home in candidate_homes(settings):
        if not home.exists():
            continue
        # Portable zip extracts to sumo-1.x.x/
        search = [home, *sorted(home.glob("sumo-*"))]
        for root in search:
            binary = resolve_bin("sumo", root)
            netconvert = resolve_bin("netconvert", root)
            if binary:
                version = "unknown"
                try:
                    import subprocess
                    proc = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=8, check=False)
                    line = (proc.stdout or proc.stderr or "").splitlines()
                    version = line[0].strip() if line else "unknown"
                except Exception:
                    version = "unknown"
                tools = root / "tools"
                return {
                    "installed": True,
                    "sumo_home": str(root),
                    "sumo_binary": binary,
                    "netconvert": netconvert,
                    "tools": str(tools) if tools.is_dir() else None,
                    "version": version,
                    "source": "detected",
                }
    which = resolve_bin("sumo", None)
    if which:
        return {
            "installed": True,
            "sumo_home": os.environ.get("SUMO_HOME") or str(Path(which).resolve().parents[1]),
            "sumo_binary": which,
            "netconvert": resolve_bin("netconvert", None),
            "tools": None,
            "version": "on-PATH",
            "source": "PATH",
        }
    return {
        "installed": False,
        "sumo_home": None,
        "sumo_binary": None,
        "netconvert": None,
        "tools": None,
        "version": None,
        "source": None,
        "message": "Eclipse SUMO was not found. Place a portable build in tools/sumo or set SUMO_HOME.",
    }
