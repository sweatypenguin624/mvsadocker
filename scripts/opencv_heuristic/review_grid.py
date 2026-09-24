"""Browser-based grid viewer for reviewing HSV-heuristic bus color predictions.

Classifies every crop in --crops-dir with classify_bus_crop() once at
startup, then serves a thumbnail grid with filter tabs (All/Red/Blue/
Yellow/Rest) so predictions can be eyeballed quickly. Read-only -- unlike
sort_crops.py this never moves/labels files, it's purely for reviewing how
the heuristic (and any config.py tuning) is doing.

Deliberately stdlib-only, same reasoning as sort_crops.py: may be run with a
bare system python3 outside the venv.

Usage:
    env/bin/python scripts/opencv_heuristic/review_grid.py \\
        --crops-dir results/bus_color/unlabeled_crop

Then, on your Mac:
    ssh -L 8767:localhost:8767 oauser@<host>
    open http://localhost:8767
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

import cv2  # noqa: E402

from classify import classify_bus_crop  # noqa: E402
from config import PipelineConfig  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_ORDER = ["red", "blue", "yellow", "rest"]

HTML = r"""
<!DOCTYPE html>
<html>
<head>
    <title>MVSA Bus Color Review</title>
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; }
        #top { padding: 10px 16px; background: #1c1c1c; position: sticky; top: 0; z-index: 5;
               display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
        #top b { margin-right: 8px; }
        .tab { padding: 8px 16px; border-radius: 6px; background: #2a2a2a; cursor: pointer;
               border: 2px solid transparent; user-select: none; }
        .tab:hover { background: #3a3a3a; }
        .tab.active { border-color: #0af; background: #14324a; }
        .tab .n { opacity: 0.7; margin-left: 6px; }
        .c-all { color: #eee; } .c-red { color: #f66; } .c-blue { color: #6af; }
        .c-yellow { color: #fd5; } .c-rest { color: #aaa; }
        #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 6px; padding: 10px; }
        .cell { position: relative; background: #000; border-radius: 4px; overflow: hidden;
                cursor: pointer; aspect-ratio: 4/3; }
        .cell img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .cell .tag { position: absolute; bottom: 0; left: 0; right: 0; font-size: 11px;
                     padding: 2px 4px; background: rgba(0,0,0,0.65); display: flex; justify-content: space-between; }
        .cell .conf { opacity: 0.75; }
        #empty { text-align: center; padding: 60px; color: #777; display: none; }
        #modal { position: fixed; inset: 0; background: rgba(0,0,0,0.9); display: none;
                 align-items: center; justify-content: center; flex-direction: column; z-index: 10; }
        #modal img { max-width: 92vw; max-height: 80vh; }
        #modal-info { margin-top: 10px; color: #ccc; font-family: monospace; }
        #modal-close { position: absolute; top: 16px; right: 24px; font-size: 28px; cursor: pointer; color: #ccc; }
    </style>
</head>
<body>
    <div id="top">
        <b>Bus Color Review</b>
        <div class="tab active c-all" data-c="all">All<span class="n" id="n-all"></span></div>
        <div class="tab c-red" data-c="red">Red<span class="n" id="n-red"></span></div>
        <div class="tab c-blue" data-c="blue">Blue<span class="n" id="n-blue"></span></div>
        <div class="tab c-yellow" data-c="yellow">Yellow<span class="n" id="n-yellow"></span></div>
        <div class="tab c-rest" data-c="rest">Rest<span class="n" id="n-rest"></span></div>
    </div>
    <div id="grid"></div>
    <div id="empty">No crops in this category.</div>

    <div id="modal">
        <span id="modal-close">&times;</span>
        <img id="modal-img" src="">
        <div id="modal-info"></div>
    </div>

<script>
let DATA = null;   // {red: [[name,conf],...], blue: [...], yellow: [...], rest: [...]}
let current = "all";

async function load() {
    const res = await fetch("/api/data");
    DATA = await res.json();
    let total = 0;
    for (const c of ["red", "blue", "yellow", "rest"]) {
        document.getElementById("n-" + c).textContent = " (" + DATA[c].length + ")";
        total += DATA[c].length;
    }
    document.getElementById("n-all").textContent = " (" + total + ")";
    render();
}

function render() {
    const grid = document.getElementById("grid");
    grid.innerHTML = "";
    let items = [];
    if (current === "all") {
        for (const c of ["red", "blue", "yellow", "rest"]) {
            for (const [name, conf] of DATA[c]) items.push([name, conf, c]);
        }
    } else {
        for (const [name, conf] of DATA[current]) items.push([name, conf, current]);
    }
    document.getElementById("empty").style.display = items.length ? "none" : "block";
    const frag = document.createDocumentFragment();
    for (const [name, conf, c] of items) {
        const cell = document.createElement("div");
        cell.className = "cell";
        const img = document.createElement("img");
        img.src = "/img/" + encodeURIComponent(name);
        img.loading = "lazy";
        const tag = document.createElement("div");
        tag.className = "tag";
        tag.innerHTML = '<span class="c-' + c + '">' + c + '</span><span class="conf">' +
                         (conf === null ? "-" : conf.toFixed(2)) + '</span>';
        cell.appendChild(img);
        cell.appendChild(tag);
        cell.onclick = () => openModal(name, conf, c);
        frag.appendChild(cell);
    }
    grid.appendChild(frag);
}

function openModal(name, conf, c) {
    document.getElementById("modal-img").src = "/img/" + encodeURIComponent(name);
    document.getElementById("modal-info").textContent =
        name + "  |  " + c + "  |  conf=" + (conf === null ? "-" : conf.toFixed(3));
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


def classify_all(crops_dir: Path, cfg: PipelineConfig):
    paths = sorted(p for p in crops_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    by_class = {c: [] for c in CLASS_ORDER}
    for p in paths:
        crop = cv2.imread(str(p))
        color, conf = classify_bus_crop(crop, cfg)
        if color is None:
            color = cfg.FALLBACK_COLOR
            conf = None
        by_class.setdefault(color, []).append([p.name, conf])
    return by_class


def make_handler(crops_dir: Path, data: dict):
    data_json = json.dumps(data).encode()

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
                name = unquote(self.path[len("/img/"):])
                path = crops_dir / name
                if ".." in name or not path.is_file():
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
    p.add_argument("--crops-dir", type=Path, required=True, help="Dir of bus crop images to classify + review.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8767)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.crops_dir.is_dir():
        raise SystemExit(f"--crops-dir not found: {args.crops_dir}")

    cfg = PipelineConfig()
    print(f"Classifying crops in {args.crops_dir} ...")
    data = classify_all(args.crops_dir, cfg)
    for c in CLASS_ORDER:
        print(f"  {c}: {len(data[c])}")

    import socket
    hostname = socket.gethostname()
    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"On your Mac:\n  ssh -L {args.port}:localhost:{args.port} oauser@{hostname}\n  open http://localhost:{args.port}")

    server = HTTPServer((args.host, args.port), make_handler(args.crops_dir, data))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
