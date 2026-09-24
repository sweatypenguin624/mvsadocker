"""YOLO detector loading and bus-class-id resolution."""

from __future__ import annotations

from ultralytics import YOLO


def load_detector(model_name: str):
    print("Loading detector...")
    detector = YOLO(model_name)
    print("Detector loaded.")
    print("Detector classes:")
    for k, v in detector.names.items():
        print(f"  {k}: {v}")
    return detector


def resolve_bus_class_id(detector, bus_class_id: int | None) -> int:
    names = detector.names
    if bus_class_id is None:
        for cid, cname in names.items():
            if str(cname).strip().lower() == "bus":
                print(f"Auto-detected BUS_CLASS_ID = {cid} ('{cname}')")
                return cid
        print("ERROR: Could not find 'bus' in detector class names.")
        print("Available classes:", names)
        raise RuntimeError("No 'bus' class found in detector. See printed classes above.")
    if bus_class_id not in names:
        raise RuntimeError(f"BUS_CLASS_ID={bus_class_id} not found in detector.names: {names}")
    print(f"Using configured BUS_CLASS_ID = {bus_class_id} ('{names[bus_class_id]}')")
    return bus_class_id


def resolve_class_ids(detector, class_names: list) -> list:
    """Resolve a list of class names (e.g. ["truck", "car"]) to detector
    class ids, skipping (with a warning) any name not present in this
    detector's classes."""
    by_name = {str(v).strip().lower(): k for k, v in detector.names.items()}
    ids = []
    for name in class_names:
        cid = by_name.get(name.strip().lower())
        if cid is None:
            print(f"[WARN] class '{name}' not found in detector.names; skipping it.")
            continue
        ids.append(cid)
    return ids
