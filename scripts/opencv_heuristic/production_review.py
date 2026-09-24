"""Browser-based grid viewer for a run_production.py output directory.

Unlike review_grid.py (which re-classifies a raw crops folder on the fly),
this reads a production run's already-computed results directly:
best_frames/<color>/*.jpg plus track_details.csv for per-track metadata
(camera, video, track id, best-frame wallclock, detection confidence, color
confidence). Read-only, stdlib-only web server (same reasoning as
sort_crops.py/review_grid.py -- may be run outside the venv... except this
one needs pandas to read the CSV, so use env/bin/python).

Usage:
    env/bin/python scripts/opencv_heuristic/production_review.py \\
        --run-dir results/bus_color/production_demo_run

Then, on your Mac:
    ssh -L 8768:localhost:8768 oauser@<host>
    open http://localhost:8768
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

CLASS_ORDER = ["red", "blue", "yellow", "rest"]

HTML = r"""
<!DOCTYPE html>
<html>
<head>
    <title>MVSA Production Run Review</title>
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; }
        #top { padding: 10px 16px; background: #1c1c1c; position: sticky; top: 0; z-index: 5;
               display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
        #top b { margin-right: 8px; }
        #meta { color: #888; font-size: 12px; margin-left: auto; }
        .tab { padding: 8px 16px; border-radius: 6px; background: #2a2a2a; cursor: pointer;
               border: 2px solid transparent; user-select: none; }
        .tab:hover { background: #3a3a3a; }
        .tab.active { border-color: #0af; background: #14324a; }
        .tab .n { opacity: 0.7; margin-left: 6px; }
        .c-all { color: #eee; } .c-red { color: #f66; } .c-blue { color: #6af; }
        .c-yellow { color: #fd5; } .c-rest { color: #aaa; }
        #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
                gap: 6px; padding: 10px; }
        .cell { position: relative; background: #000; border-radius: 4px; overflow: hidden;
                cursor: pointer; aspect-ratio: 4/3; }
        .cell img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .cell .tag { position: absolute; bottom: 0; left: 0; right: 0; font-size: 11px;
                     padding: 2px 4px; background: rgba(0,0,0,0.65); display: flex; justify-content: space-between; }
        .cell .cam { position: absolute; top: 0; left: 0; right: 0; font-size: 10px;
                     padding: 2px 4px; background: rgba(0,0,0,0.55); color: #9cf; white-space: nowrap;
                     overflow: hidden; text-overflow: ellipsis; }
        #empty { text-align: center; padding: 60px; color: #777; display: none; }
        #modal { position: fixed; inset: 0; background: rgba(0,0,0,0.9); display: none;
                 align-items: center; justify-content: center; flex-direction: column; z-index: 10; }
        #modal img { max-width: 92vw; max-height: 75vh; }
        #modal-info { margin-top: 10px; color: #ccc; font-family: monospace; font-size: 13px; text-align: center; }
        #modal-close { position: absolute; top: 16px; right: 24px; font-size: 28px; cursor: pointer; color: #ccc; }
    </style>
</head>
<body>
    <div id="top">
        <b>Production Run Review</b>
        <div class="tab active c-all" data-c="all">All<span class="n" id="n-all"></span></div>
        <div class="tab c-red" data-c="red">Red<span class="n" id="n-red"></span></div>
        <div class="tab c-blue" data-c="blue">Blue<span class="n" id="n-blue"></span></div>
        <div class="tab c-yellow" data-c="yellow">Yellow<span class="n" id="n-yellow"></span></div>
        <div class="tab c-rest" data-c="rest">Rest<span class="n" id="n-rest"></span></div>
        <div id="meta"></div>
    </div>
    <div id="grid"></div>
    <div id="empty">No tracks in this category.</div>

    <div id="modal">
        <span id="modal-close">&times;</span>
        <img id="modal-img" src="">
        <div id="modal-info"></div>
    </div>

<script>
let DATA = null;   // {red: [track,...], blue: [...], yellow: [...], rest: [...]}
let current = "all";

async function load() {
    const res = await fetch("/api/data");
    DATA = await res.json();
    let total = 0;
    for (const c of ["red", "blue", "yellow", "rest"]) {
        document.getElementById("n-" + c).textContent = " (" + DATA.tracks[c].length + ")";
        total += DATA.tracks[c].length;
    }
    document.getElementById("n-all").textContent = " (" + total + ")";
    document.getElementById("meta").textContent = DATA.run_dir + " -- " + DATA.videos.join(", ");
    render();
}

function render() {
    const grid = document.getElementById("grid");
    grid.innerHTML = "";
    let items = [];
    if (current === "all") {
        for (const c of ["red", "blue", "yellow", "rest"]) {
            for (const t of DATA.tracks[c]) items.push(t);
        }
    } else {
        items = DATA.tracks[current];
    }
    document.getElementById("empty").style.display = items.length ? "none" : "block";
    const frag = document.createDocumentFragment();
    for (const t of items) {
        const cell = document.createElement("div");
        cell.className = "cell";
        const img = document.createElement("img");
        img.src = "/img/" + encodeURIComponent(t.image);
        img.loading = "lazy";
        const cam = document.createElement("div");
        cam.className = "cam";
        cam.textContent = t.camera;
        const tag = document.createElement("div");
        tag.className = "tag";
        tag.innerHTML = '<span class="c-' + t.color + '">' + t.color + '</span><span>' + t.time + '</span>';
        cell.appendChild(cam);
        cell.appendChild(img);
        cell.appendChild(tag);
        cell.onclick = () => openModal(t);
        frag.appendChild(cell);
    }
    grid.appendChild(frag);
}

function openModal(t) {
    document.getElementById("modal-img").src = "/img/" + encodeURIComponent(t.image);
    document.getElementById("modal-info").textContent =
        `camera=${t.camera}  video=${t.video}  track_id=${t.track_id}\n` +
        `best_frame=${t.time}  detection_conf=${t.det_conf}  color=${t.color}  color_conf=${t.color_conf}`;
    document.getElementById("modal-info").style.whiteSpace = "pre";
    document.getElementById("modal").style.display = "flex";
}
document.getElementById("modal-close").onclick = () => document.getElementById("modal").style.display = "none";
document.getElementById("modal").onclick = (e) => { if (e.target.id === "modal") e.currentTarget.style.display = "none"; };

for (const tab of document.querySelectorAll(".tab")) {
    tab.onclick = () => {
        document.querySelector(".tab.active").classList.remove("active");
        tab.classList.add("active");
        current = tab.dataset.c;
        render();
    };
}

load();
</script>
</body>
</html>
"""


def load_run_data(run_dir: Path) -> dict:
    csv_path = run_dir / "track_details.csv"
    if not csv_path.is_file():
        raise SystemExit(f"No track_details.csv found under {run_dir} -- is this a run_production.py output dir?")
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    best_dir = run_dir / "best_frames"
    # image filenames are "<camera>_<video_stem>_track<NNNNN>_<color>.jpg" (see mode_production.py)
    image_by_key = {}
    for color in CLASS_ORDER:
        color_dir = best_dir / color
        if not color_dir.is_dir():
            continue
        for p in color_dir.iterdir():
            if p.is_file():
                image_by_key[(color, p.name)] = p

    tracks = {c: [] for c in CLASS_ORDER}
    for _, row in df.iterrows():
        color = row["color"] if row["color"] in CLASS_ORDER else "rest"
        video_stem = Path(row["video"]).stem
        track_id = int(row["track_id"]) if row["track_id"] else 0
        expected_name = f"{row['camera']}_{video_stem}_track{track_id:05d}_{color}.jpg"
        image_path = best_dir / color / expected_name
        if not image_path.is_file():
            continue  # track had no valid best frame (e.g. crop never passed crop_bus())
        tracks[color].append({
            "image": f"{color}/{expected_name}",
            "camera": row["camera"],
            "video": row["video"],
            "track_id": track_id,
            "time": row["best_frame_wallclock"].replace("T", " "),
            "det_conf": row["best_frame_detection_confidence"],
            "color": color,
            "color_conf": row["color_confidence"],
        })
    for c in CLASS_ORDER:
        tracks[c].sort(key=lambda t: t["time"])

    return {
        "run_dir": str(run_dir),
        "videos": sorted(df["video"].unique().tolist()),
        "tracks": tracks,
    }


def make_handler(run_dir: Path, data: dict):
    data_json = json.dumps(data).encode()
    best_dir = run_dir / "best_frames"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                body = HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/data":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data_json)))
                self.end_headers()
                self.wfile.write(data_json)
            elif self.path.startswith("/img/"):
                rel = unquote(self.path[len("/img/"):])
                path = best_dir / rel
                if ".." in rel or not path.is_file():
                    self.send_response(404)
                    self.end_headers()
                    return
                img_bytes = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(img_bytes)))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(img_bytes)
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="A run_production.py output directory.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8768)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.run_dir.is_dir():
        raise SystemExit(f"--run-dir not found: {args.run_dir}")

    data = load_run_data(args.run_dir)
    for c in CLASS_ORDER:
        print(f"  {c}: {len(data['tracks'][c])}")

    import socket
    hostname = socket.gethostname()
    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"On your Mac:\n  ssh -L {args.port}:localhost:{args.port} oauser@{hostname}\n  open http://localhost:{args.port}")

    server = HTTPServer((args.host, args.port), make_handler(args.run_dir, data))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
