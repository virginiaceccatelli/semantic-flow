"""Run manifests: every pipeline stage records what ran, on what, from where."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MANIFEST_DIR = Path("results/manifests")


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def write_manifest(stage: str, args: dict, started_at: float, extra: dict | None = None) -> Path:
    """Write a JSON manifest for one stage run; returns its path."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {
        "stage": stage,
        "timestamp": ts,
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "argv": sys.argv,
        "args": {k: str(v) for k, v in args.items()},
        "wall_time_s": round(time.time() - started_at, 1),
    }
    if extra:
        manifest["extra"] = extra
    path = MANIFEST_DIR / f"{stage}_{ts}.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
