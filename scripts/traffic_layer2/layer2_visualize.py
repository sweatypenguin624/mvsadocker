"""Debug text visualization for Layer 2 results (spec section 24)."""

from __future__ import annotations

from typing import List

from layer2_models import Layer2Result


def format_track(result: Layer2Result) -> str:
    lines = [
        f"Track: {result.track_id}",
        f"Layer 1: {result.layer1_class}",
        f"Layer 2: {result.layer2_class}",
        f"Confidence: {result.confidence:.2f}",
        f"Direction: {result.direction}",
    ]
    if result.branch_path:
        lines.append(f"Branch: {' -> '.join(result.branch_path)}")
    if result.trolley_track_id is not None:
        lines.append(f"Trolley track: {result.trolley_track_id}")
    if result.is_uncertain:
        lines.append("(uncertain -- excluded from confident subclass counts per config)")
    return "\n".join(lines)


def write_debug_report(results: List[Layer2Result], output_path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(format_track(r) + "\n\n")
