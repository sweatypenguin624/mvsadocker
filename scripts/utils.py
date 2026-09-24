"""Generic, dependency-light helpers: logging setup, weighted statistics,
and an environment sanity check. Nothing here touches CUDA, YOLO or
InsightFace, so it stays importable (and testable) without a GPU.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing, and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(log_path: Path, level: str = "INFO", name: str = "mvsa") -> logging.Logger:
    """Configure and return the shared MVSA logger.

    Logs to both stderr and ``log_path``. Safe to call multiple times
    (handlers are not duplicated).
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ensure_dir(log_path.parent)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    """Compute a weighted median.

    Robust to outliers in a way a plain mean is not -- a single wildly wrong
    age estimate on one frame does not skew the track-level result the way
    it would with an arithmetic mean.

    Returns None if ``values`` is empty or all weights are zero.
    """
    if not values:
        return None
    pairs = sorted(zip(values, weights), key=lambda vw: vw[0])
    total_weight = sum(w for _, w in pairs)
    if total_weight <= 0:
        # Fall back to an unweighted median so a sample list with all-zero
        # weights still produces a usable answer instead of silently
        # discarding valid measurements.
        pairs = sorted(zip(values, [1.0] * len(values)), key=lambda vw: vw[0])
        total_weight = float(len(pairs))

    cumulative = 0.0
    half = total_weight / 2.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= half:
            return value
    return pairs[-1][0]


def weighted_majority_vote(
    labels: Sequence[str], weights: Sequence[float]
) -> Tuple[Optional[str], float]:
    """Pick the label with the highest total weight.

    Returns ``(label, total_weight_fraction)`` where the fraction is that
    label's share of total weight (useful as a confidence proxy). Returns
    ``(None, 0.0)`` for an empty input.
    """
    if not labels:
        return None, 0.0
    totals = {}
    for label, weight in zip(labels, weights):
        totals[label] = totals.get(label, 0.0) + max(weight, 0.0)
    total_weight = sum(totals.values())
    if total_weight <= 0:
        return None, 0.0
    best_label = max(totals, key=totals.get)
    return best_label, totals[best_label] / total_weight


def verify_core_dependencies(logger: logging.Logger) -> None:
    """Warn (do not fail) if torch/torchvision/numpy differ from the known-
    good cluster versions. This is a safety net for the "do not break the
    existing environment" requirement -- it never aborts the run, since a
    developer laptop or CI box legitimately may have different versions.
    """
    expected = {
        "torch": "2.1.0+cu118",
        "torchvision": "0.16.0+cu118",
        "numpy": "1.26.1",
    }
    for module_name, expected_version in expected.items():
        try:
            module = __import__(module_name)
            actual = getattr(module, "__version__", "unknown")
        except ImportError:
            logger.warning("Dependency check: %s is not importable", module_name)
            continue
        if actual != expected_version:
            logger.warning(
                "Dependency check: %s version is %s, expected %s "
                "(the cluster's working environment). If this run is on the "
                "research server, verify nothing (e.g. pip installing "
                "insightface without --no-deps) upgraded it.",
                module_name,
                actual,
                expected_version,
            )
        else:
            logger.info("Dependency check: %s %s OK", module_name, actual)
