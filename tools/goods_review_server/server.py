#!/usr/bin/env python3
"""Local web UI to review/correct Layer 2 Goods Vehicle subclassifications.

Reads <layer2_dir>/tracks_layer2.jsonl (written by
scripts/traffic_layer2/classify_goods.py) and each track's crops from
<layer2_dir>/goods_tracks/<id>/. Corrections are saved to
<layer2_dir>/corrections.jsonl (append-only, latest correction per
track_id wins) -- never overwrites the original model output, so the raw
Layer 2 predictions stay intact for later comparison/training.

Usage:
    python tools/goods_review_server/server.py --layer2-dir results/<run>/layer2 --port 8766
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

GOODS_TAXONOMY_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "layer2_goods_taxonomy.yaml"
BUS_TAXONOMY_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "layer2_bus_taxonomy.yaml"


def load_subclasses() -> list:
    import yaml
    with open(GOODS_TAXONOMY_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    subclasses = list(raw.get("subclasses", []))
    subclasses.append(raw.get("uncertain", {}).get("heavy_truck_label", "Uncertain Heavy Truck"))
    return subclasses


def load_bus_subclasses() -> list:
    import yaml
    with open(BUS_TAXONOMY_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    subclasses = list(raw.get("subclasses", []))
    subclasses.append(raw.get("uncertain", {}).get("label", "Uncertain Bus Type"))
    return subclasses


# Layer 1's fine-grained UVH-26 detector sometimes emits Truck/LCV/
# tempo-traveller for a physical Bus or Three-wheeler (tiled/overlapping
# boxes, close-range misclassification -- see the Layer 1 accuracy-check
# memory notes) or ByteTrack briefly stitches the wrong crop onto a track.
# The reviewer needs an escape hatch to say "this was never a Goods
# Vehicle" rather than being forced to pick one of the 8 subclasses for a
# crop that's plainly a bus or auto -- kept as its own labeled group in the
# UI, and excluded from goods-vehicle totals/dataset export the same way
# "Uncertain Heavy Truck" already is.
NON_GOODS_RECLASSIFY_LABELS = [
    "Not Goods Vehicle: Bus",
    "Not Goods Vehicle: Auto/Three-wheeler",
    "Not Goods Vehicle: Car",
    "Not Goods Vehicle: 2-Wheeler",
    "Not Goods Vehicle: Cycle",
    "Not Goods Vehicle: Other",
]

# Correction options for the passthrough classes (Two-wheeler, Car-shaped
# tracks, etc.) that Layer 2 doesn't subclassify -- UVH-26 already reports
# these natively, but a reviewer going through every track needs a way to
# fix a wrong native class without being forced into the Goods/Bus label
# sets. "Truck (unspecified axle)" exists because judging exact axle count
# from a single photo is unreliable for a human reviewer too -- it's a
# valid, deliberately coarser answer than the 8-class Goods taxonomy's
# 2/3-Axle/MAV split, not a stand-in for it.
BROAD_VEHICLE_LABELS = [
    "Two-Wheeler",
    "Car",
    "Truck (unspecified axle)",
]


class ReviewState:
    def __init__(self, layer2_dir: Path):
        self.layer2_dir = layer2_dir
        self.tracks_path = layer2_dir / "tracks_layer2.jsonl"
        self.corrections_path = layer2_dir / "corrections.jsonl"
        self.subclasses = load_subclasses()
        self.bus_subclasses = load_bus_subclasses()
        self.non_goods_labels = NON_GOODS_RECLASSIFY_LABELS
        self.broad_vehicle_labels = BROAD_VEHICLE_LABELS
        self.all_labels = self.subclasses + self.bus_subclasses + self.broad_vehicle_labels + self.non_goods_labels

    def load_tracks(self) -> list:
        tracks = []
        if not self.tracks_path.exists():
            return tracks
        with open(self.tracks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tracks.append(json.loads(line))

        corrections = self.load_corrections()
        for t in tracks:
            corr = corrections.get(str(t["track_id"]))
            t["corrected_label"] = corr["corrected_label"] if corr else None
            t["corrected_at"] = corr["corrected_at"] if corr else None
        return tracks

    def load_corrections(self) -> dict:
        latest = {}
        if not self.corrections_path.exists():
            return latest
        with open(self.corrections_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                latest[str(rec["track_id"])] = rec  # later lines overwrite earlier -> latest wins
        return latest

    def save_correction(self, track_id: int, corrected_label: str, note: str = "") -> None:
        if corrected_label not in self.all_labels:
            raise ValueError(f"unknown label {corrected_label!r}")
        record = {
            "track_id": track_id,
            "corrected_label": corrected_label,
            "note": note,
            "corrected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with open(self.corrections_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Goods Vehicle Layer 2 Review</title>
<style>
  body { font-family: -apple-system, sans-serif; background: #14161a; color: #e6e6e6; margin: 0; padding: 20px; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  #stats { color: #999; margin-bottom: 16px; font-size: 13px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }
  .card { background: #1e2126; border-radius: 8px; padding: 10px; border: 1px solid #2c2f36; }
  .card.corrected { border-color: #3a8f5f; }
  .card img { width: 100%; border-radius: 4px; background: #000; aspect-ratio: 4/3; object-fit: contain; }
  .row { display: flex; justify-content: space-between; font-size: 12px; color: #aaa; margin-top: 6px; }
  .pred { font-weight: 600; font-size: 14px; margin-top: 6px; color: #f0c674; }
  .conf { color: #7fb0e0; }
  select { width: 100%; margin-top: 8px; padding: 6px; background: #0e1013; color: #eee; border: 1px solid #333; border-radius: 4px; }
  .save-btn { width: 100%; margin-top: 6px; padding: 6px; background: #3a5f8f; color: white; border: none; border-radius: 4px; cursor: pointer; }
  .save-btn:hover { background: #4a75ab; }
  .corrected-badge { color: #3a8f5f; font-size: 11px; margin-top: 4px; }
  .uncertain { color: #d97757; }
  select.uncertain-select { background: #331a12; }
</style>
</head>
<body>
<h1>Goods Vehicle Layer 2 Review</h1>
<div id="stats">Loading... &nbsp;|&nbsp; <a href="/annotate" style="color:#7fb0e0;">Quick Annotate mode &rarr;</a></div>
<div class="grid" id="grid"></div>
<script>
async function load() {
  const res = await fetch('/api/tracks');
  const tracks = await res.json();
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  let correctedCount = 0;
  tracks.forEach(t => { if (t.corrected_label) correctedCount++; });
  document.getElementById('stats').innerHTML =
    `${tracks.length} track(s) -- ${correctedCount} corrected, ${tracks.length - correctedCount} pending`
    + ` &nbsp;|&nbsp; <a href="/annotate" style="color:#7fb0e0;">Quick Annotate mode &rarr;</a>`;

  tracks.forEach(t => {
    const card = document.createElement('div');
    card.className = 'card' + (t.corrected_label ? ' corrected' : '');
    const isUncertain = t.layer_2_class.startsWith('Uncertain');

    const img = document.createElement('img');
    img.src = `/crop/${t.track_id}`;
    img.alt = `track ${t.track_id}`;
    card.appendChild(img);

    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = `<span>Track ${t.track_id}</span><span>${t.branch_path.join(' \\u2192 ')}</span>`;
    card.appendChild(row);

    const pred = document.createElement('div');
    pred.className = 'pred' + (isUncertain ? ' uncertain' : '');
    pred.innerHTML = `${t.layer_2_class} <span class="conf">(${(t.confidence*100).toFixed(0)}%)</span>`;
    card.appendChild(pred);

    const select = document.createElement('select');
    if (isUncertain) select.className = 'uncertain-select';
    const currentValue = t.corrected_label || t.layer_2_class;

    const goodsGroup = document.createElement('optgroup');
    goodsGroup.label = 'Goods Vehicle subclass';
    window.GOODS_LABELS.forEach(label => {
      const opt = document.createElement('option');
      opt.value = label;
      opt.textContent = label;
      if (currentValue === label) opt.selected = true;
      goodsGroup.appendChild(opt);
    });
    select.appendChild(goodsGroup);

    const busGroup = document.createElement('optgroup');
    busGroup.label = 'Bus subclass';
    window.BUS_LABELS.forEach(label => {
      const opt = document.createElement('option');
      opt.value = label;
      opt.textContent = label;
      if (currentValue === label) opt.selected = true;
      busGroup.appendChild(opt);
    });
    select.appendChild(busGroup);

    const broadGroup = document.createElement('optgroup');
    broadGroup.label = 'Vehicle type (broad)';
    window.BROAD_VEHICLE_LABELS.forEach(label => {
      const opt = document.createElement('option');
      opt.value = label;
      opt.textContent = label;
      if (currentValue === label) opt.selected = true;
      broadGroup.appendChild(opt);
    });
    select.appendChild(broadGroup);

    const nonGoodsGroup = document.createElement('optgroup');
    nonGoodsGroup.label = 'Not actually a Goods Vehicle';
    window.NON_GOODS_LABELS.forEach(label => {
      const opt = document.createElement('option');
      opt.value = label;
      opt.textContent = label;
      if (currentValue === label) opt.selected = true;
      nonGoodsGroup.appendChild(opt);
    });
    select.appendChild(nonGoodsGroup);

    card.appendChild(select);

    const btn = document.createElement('button');
    btn.className = 'save-btn';
    btn.textContent = 'Save correction';
    btn.onclick = async () => {
      btn.textContent = 'Saving...';
      await fetch('/api/correct', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({track_id: t.track_id, corrected_label: select.value})
      });
      btn.textContent = 'Saved';
      card.classList.add('corrected');
      setTimeout(() => { btn.textContent = 'Save correction'; }, 1200);
    };
    card.appendChild(btn);

    if (t.corrected_label) {
      const badge = document.createElement('div');
      badge.className = 'corrected-badge';
      badge.textContent = `Corrected -> ${t.corrected_label}`;
      card.appendChild(badge);
    }

    grid.appendChild(card);
  });
}
load();
</script>
</body>
</html>
"""


ANNOTATE_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Quick Annotate</title>
<style>
  body { font-family: -apple-system, sans-serif; background: #0e1013; color: #e6e6e6; margin: 0;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  #topbar { padding: 10px 20px; display: flex; justify-content: space-between; align-items: center;
            border-bottom: 1px solid #2c2f36; font-size: 13px; color: #999; }
  #topbar a { color: #7fb0e0; }
  #stage { flex: 1; display: flex; align-items: center; justify-content: center; position: relative; min-height: 0; }
  #bigimg { max-width: 92vw; max-height: 78vh; object-fit: contain; border-radius: 6px; background: #000; }
  #label-badge { position: absolute; top: 16px; right: 16px; padding: 6px 14px; border-radius: 20px;
                 font-weight: 600; font-size: 14px; background: #1e2126; border: 1px solid #3a8f5f; color: #3a8f5f; }
  #label-badge.hidden { display: none; }
  #meta { text-align: center; padding: 8px; color: #999; font-size: 13px; }
  #footer { padding: 12px 20px; border-top: 1px solid #2c2f36; display: flex; justify-content: center;
            gap: 24px; font-size: 13px; color: #ccc; }
  kbd { background: #2c2f36; border-radius: 4px; padding: 2px 7px; font-family: monospace; margin-right: 6px; }
  #toast { position: fixed; bottom: 70px; left: 50%; transform: translateX(-50%); background: #3a8f5f;
           color: white; padding: 8px 18px; border-radius: 20px; font-size: 13px; opacity: 0; transition: opacity 0.3s; pointer-events: none; }
  #toast.show { opacity: 1; }
</style>
</head>
<body>
<div id="topbar">
  <span id="progress">Loading...</span>
  <a href="/">&larr; grid view</a>
</div>
<div id="stage">
  <div id="label-badge" class="hidden"></div>
  <img id="bigimg" src="" alt="">
</div>
<div id="meta">Track --</div>
<div id="footer"></div>
<div id="toast"></div>
<script>
// key -> label, in the order shown in the footer. Extend this to add more
// quick-annotate keys later; anything not covered here still has the full
// dropdown in grid view (/).
// value = what actually gets saved (matches the taxonomy files);
// display = what the footer shows, since "Goods 3 Wheeler" alone reads as
// unfamiliar/missing to a reviewer thinking "auto rickshaw".
const KEY_MAP = {
  '1': { value: 'Goods 3 Wheeler', display: 'Auto Rickshaw / Goods 3-Wheeler' },
  '2': { value: 'BRTC Bus', display: 'BRTC Bus' },
  '3': { value: 'City/Private Bus', display: 'City/Private Bus' },
  '4': { value: 'School Bus', display: 'School Bus' },
  '5': { value: 'Two-Wheeler', display: 'Two-Wheeler' },
  '6': { value: 'Car', display: 'Car (Hatchback/Sedan/SUV/MUV/Van)' },
  '7': { value: 'Truck (unspecified axle)', display: 'Truck (broad, no axle guess)' },
  '8': { value: 'LCV', display: 'LCV' },
  '9': { value: 'Tractor', display: 'Tractor' },
};
let tracks = [];
let idx = 0;
let annotatedThisSession = 0;

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 1400);
}

function render() {
  if (!tracks.length) return;
  idx = Math.max(0, Math.min(tracks.length - 1, idx));
  const t = tracks[idx];
  document.getElementById('bigimg').src = `/crop/${t.track_id}`;
  document.getElementById('meta').textContent =
    `Track ${t.track_id}  --  Layer 2 predicted: ${t.layer_2_class} (${(t.confidence*100).toFixed(0)}%)`;
  const badge = document.getElementById('label-badge');
  if (t.corrected_label) {
    badge.textContent = t.corrected_label + ' \\u2713';
    badge.classList.remove('hidden');
  } else {
    badge.classList.add('hidden');
  }
  const labeledCount = tracks.filter(x => x.corrected_label).length;
  document.getElementById('progress').textContent =
    `${idx + 1} / ${tracks.length}   --   ${labeledCount} labeled total`;
  localStorage.setItem('annotate_idx', String(idx));
}

function advance(delta) {
  idx += delta;
  if (idx >= tracks.length) idx = tracks.length - 1;
  if (idx < 0) idx = 0;
  render();
}

async function saveLabel(label) {
  const t = tracks[idx];
  t.corrected_label = label;  // optimistic local update, saved below
  render();
  fetch('/api/correct', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({track_id: t.track_id, corrected_label: label})
  });
  annotatedThisSession++;
  if (annotatedThisSession % 20 === 0) {
    showToast(`Saved -- ${annotatedThisSession} labeled this session`);
  }
  advance(1);
}

document.addEventListener('keydown', (e) => {
  if (KEY_MAP[e.key]) {
    saveLabel(KEY_MAP[e.key].value);
  } else if (e.key === 'ArrowRight' || e.key.toLowerCase() === 'n') {
    advance(1);
  } else if (e.key === 'ArrowLeft' || e.key.toLowerCase() === 'p') {
    advance(-1);
  }
});

function renderFooter() {
  const footer = document.getElementById('footer');
  const parts = Object.entries(KEY_MAP).map(([key, info]) => `<span><kbd>${key}</kbd> ${info.display}</span>`);
  parts.push('<span><kbd>&larr;</kbd> Previous</span>');
  parts.push('<span><kbd>&rarr;</kbd> Next (skip / accept default)</span>');
  footer.innerHTML = parts.join('');
}

async function load() {
  renderFooter();
  const res = await fetch('/api/tracks');
  tracks = await res.json();
  const saved = parseInt(localStorage.getItem('annotate_idx') || '0', 10);
  idx = isNaN(saved) ? 0 : saved;
  render();
}
load();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    state: ReviewState = None  # set by main()

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = INDEX_HTML.replace(
                "<script>\nasync function load()",
                f"<script>\nwindow.GOODS_LABELS = {json.dumps(self.state.subclasses)};\n"
                f"window.BUS_LABELS = {json.dumps(self.state.bus_subclasses)};\n"
                f"window.BROAD_VEHICLE_LABELS = {json.dumps(self.state.broad_vehicle_labels)};\n"
                f"window.NON_GOODS_LABELS = {json.dumps(self.state.non_goods_labels)};\nasync function load()",
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/annotate":
            body = ANNOTATE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/tracks":
            self._send_json(self.state.load_tracks())
            return

        if parsed.path.startswith("/crop/"):
            track_id = parsed.path.split("/crop/")[1].split("/")[0]
            crop_path = self.state.layer2_dir / "goods_tracks" / f"{int(track_id):06d}" / "best.jpg"
            if not crop_path.exists():
                self.send_response(404)
                self.end_headers()
                return
            data = crop_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/correct":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            try:
                self.state.save_correction(int(body["track_id"]), body["corrected_label"], body.get("note", ""))
            except ValueError as e:
                self._send_json({"ok": False, "error": str(e)}, status=400)
                return
            self._send_json({"ok": True})
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # keep stdout quiet; corrections still visible via GET /api/tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer2-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    layer2_dir = args.layer2_dir.resolve()
    if not (layer2_dir / "tracks_layer2.jsonl").exists():
        print(f"Error: {layer2_dir / 'tracks_layer2.jsonl'} not found -- run classify_goods.py first", file=sys.stderr)
        sys.exit(1)

    Handler.state = ReviewState(layer2_dir)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Serving Layer 2 review UI for {layer2_dir}")
    print(f"Port: {args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
