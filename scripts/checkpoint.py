"""Atomic JSON checkpointing.

Deliberately generic: this module knows nothing about TrackState, FaceSample,
or any other pipeline-specific type. pipeline.py is responsible for
serializing/deserializing its own state into plain JSON-compatible dicts
(each relevant dataclass provides ``to_dict``/``from_dict`` for this). That
separation means swapping the demographic model or tracker later never
requires touching this file.

Writes are atomic (write to a temp file, then os.replace) so a process kill
mid-write can never leave a corrupted checkpoint.json on disk.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mvsa")


class CheckpointManager:
    def __init__(self, path: Path, interval_frames: int):
        self.path = Path(path)
        self.interval_frames = max(1, interval_frames)

    def should_save(self, frame_idx: int) -> bool:
        return frame_idx > 0 and frame_idx % self.interval_frames == 0

    def save(self, state: dict) -> None:
        """Atomically write ``state`` (must be JSON-serializable) to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(state)
        state["saved_at"] = datetime.now().isoformat()

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".checkpoint_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        logger.info("Checkpoint saved at frame %s -> %s", state.get("frame_idx"), self.path)

    def load(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def exists(self) -> bool:
        return self.path.exists()
