"""Per-run step timing. Stages accumulate wall-clock seconds by dotted name
(e.g. ``frame_loop.decode``) and are written as a readable .log + .json pair.
"""

from __future__ import annotations

import csv
import json
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_LOG_DIR = PROJECT_ROOT / "logs" / "runs"

_csv_lock = threading.Lock()


class RunTimer:
    def __init__(self, start: float | None = None):
        self.start = start if start is not None else time.time()
        self.stages: dict[str, list] = {}
        self.meta: dict = {}
        self.children: dict[str, dict] = {}
        self._last_lap = time.perf_counter()

    def reset_lap(self) -> None:
        self._last_lap = time.perf_counter()

    def lap(self, name: str) -> None:
        """Record time since the previous lap()/reset_lap() under ``name``."""
        now = time.perf_counter()
        self.add(name, now - self._last_lap)
        self._last_lap = now

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def add(self, name: str, seconds: float, calls: int = 1) -> None:
        s = self.stages.setdefault(name, [0.0, 0])
        s[0] += seconds
        s[1] += calls

    def seconds(self, name: str) -> float:
        return self.stages.get(name, [0.0, 0])[0]

    def to_dict(self, status: str) -> dict:
        end = time.time()
        return {
            "status": status,
            "started_at": datetime.fromtimestamp(self.start).isoformat(timespec="seconds"),
            "finished_at": datetime.fromtimestamp(end).isoformat(timespec="seconds"),
            "total_seconds": round(end - self.start, 3),
            "meta": self.meta,
            "stages": {k: {"seconds": round(v[0], 4), "calls": v[1]} for k, v in self.stages.items()},
            "children": self.children,
        }


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def _fmt_secs(s: float) -> str:
    if s >= 60:
        m, sec = divmod(s, 60)
        h, m = divmod(m, 60)
        hms = f"{int(h)}h{int(m):02d}m{sec:04.1f}s" if h else f"{int(m)}m{sec:04.1f}s"
        return f"{s:.2f}s ({hms})"
    return f"{s:.3f}s"


def _ordered(stages: dict) -> list[str]:
    # Parents sort before children even when the parent finished (was added) last.
    order: dict[str, int] = {}
    for name in stages:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            order.setdefault(".".join(parts[:i]), len(order))

    def key(name):
        parts = name.split(".")
        return tuple(order[".".join(parts[:i])] for i in range(1, len(parts) + 1))

    return sorted(stages, key=key)


def _with_group_rows(stages: dict) -> dict:
    """Add a summed row for dotted prefixes (e.g. ``setup``) that were never timed directly."""
    out = dict(stages)
    for name in stages:
        parts = name.split(".")
        for i in range(1, len(parts)):
            prefix = ".".join(parts[:i])
            if prefix not in stages and prefix not in out:
                children = [v["seconds"] for k, v in stages.items()
                            if k.startswith(prefix + ".") and k.count(".") == i]
                out[prefix] = {"seconds": sum(children), "calls": 0}
    return out


def _stage_table(stages: dict, total: float, indent: str = "") -> list[str]:
    stages = _with_group_rows(stages)
    lines = [f"{indent}{'STAGE':<44}{'TIME':>24}{'% OF TOTAL':>12}{'CALLS':>9}{'AVG/CALL':>12}"]
    for name in _ordered(stages):
        sec, calls = stages[name]["seconds"], stages[name]["calls"]
        depth = name.count(".")
        label = "  " * depth + name.split(".")[-1]
        pct = f"{100 * sec / total:.1f}%" if total > 0 else "-"
        avg = f"{1000 * sec / calls:.2f}ms" if calls > 1 else ""
        calls_s = str(calls) if calls else ""
        lines.append(f"{indent}{label:<44}{_fmt_secs(sec):>24}{pct:>12}{calls_s:>9}{avg:>12}")
    return lines


def format_report(data: dict, title: str) -> str:
    total = data["total_seconds"]
    lines = [
        "=" * 101,
        title,
        "=" * 101,
        f"status:   {data['status']}",
        f"started:  {data['started_at']}",
        f"finished: {data['finished_at']}",
        f"total:    {_fmt_secs(total)}",
        "",
        "-- metadata --",
    ]
    lines += [f"{k}: {v}" for k, v in data["meta"].items()]
    lines += ["", "-- stages --"]
    lines += _stage_table(data["stages"], total)
    for name, child in data.get("children", {}).items():
        lines += ["", f"-- {name} breakdown (% of {name} process total {_fmt_secs(child['total_seconds'])}) --"]
        lines += [f"  {k}: {v}" for k, v in child.get("meta", {}).items()]
        lines.append("")
        lines += _stage_table(child["stages"], child["total_seconds"], indent="  ")
    return "\n".join(lines) + "\n"


def write_report(base_path: Path, data: dict, title: str) -> str:
    """Write ``<base_path>.log`` and ``<base_path>.json``; return the log text."""
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    text = format_report(data, title)
    Path(f"{base_path}.log").write_text(text, encoding="utf-8")
    Path(f"{base_path}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return text


def append_summary_csv(csv_path: Path, row: dict) -> None:
    csv_path = Path(csv_path)
    with _csv_lock:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
