"""Pytest bootstrap: put scripts/ on sys.path so tests can `import roi`,
`import config_loader`, etc. exactly the way scripts/main.py does (flat
imports, no package prefix), without requiring an editable install.
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

VEHICLE_COUNTING_DIR = SCRIPTS_DIR / "vehicle_counting"
if str(VEHICLE_COUNTING_DIR) not in sys.path:
    sys.path.insert(0, str(VEHICLE_COUNTING_DIR))

TRAFFIC_LAYER1_DIR = SCRIPTS_DIR / "traffic_layer1"
if str(TRAFFIC_LAYER1_DIR) not in sys.path:
    sys.path.insert(0, str(TRAFFIC_LAYER1_DIR))
