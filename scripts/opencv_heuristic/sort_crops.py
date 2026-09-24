"""Browser-based crop sorter for bus color labeling -- keyboard-driven (1/2/3/4)
labeling UI for the crops produced by `run.py collect_crops`.

Deliberately stdlib-only (no Flask/etc.) -- same reasoning as roieditor.py:
this may be run with a bare system python3 outside the ~/mvsa/env venv, so it
avoids adding a dependency just for a labeling tool.

Usage:
    python3 scripts/opencv_heuristic/sort_crops.py \\
        --crops-dir results/bus_color/unlabeled_crops \\
        --dataset-dir results/bus_color/bus_color_dataset

Then, on your Mac:
    ssh -L 8766:localhost:8766 oauser@<host>
    open http://localhost:8766

Keys: 1 = red | 2 = blue | 3 = yellow | 4 = rest | Backspace = undo last | S = skip (leaves file unsorted)
Each classified crop is moved (not copied) from --crops-dir into
<dataset-dir>/train/<color>/, which is exactly the layout `run.py
train_classifier` expects.
"""

import argparse
import json
import os
import shutil
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

COLORS = ["red", "blue", "yellow", "rest"]
KEY_TO_COLOR = {"1": "red", "2": "blue", "3": "yellow", "4": "rest"}

HTML = r"""
<!DOCTYPE html>
<html>
<head>
    <title>MVSA Bus Color Sorter</title>
    <style>
        body { margin: 0; background: #111; color: white; font-family: Arial, sans-serif; text-align: center; }
        #top { padding: 12px 18px; background: #222; }
        #counts span { margin: 0 10px; font-weight: bold; }
        .c-red { color: #f55; } .c-blue { color: #59f; } .c-yellow { color: #fd5; } .c-rest { color: #aaa; }
        #imgwrap { margin: 24px auto; }
        #img { max-width: 90vw; max-height: 65vh; border: 2px solid #444; background: #000; }
        #name { margin-top: 10px; color: #999; font-family: monospace; }
        #remaining { margin-top: 4px; color: #0f0; }
        #keys { margin-top: 20px; font-size: 15px; }
        .key { display: inline-block; padding: 10px 18px; margin: 0 8px; border-radius: 6px; background: #333; cursor: pointer; }
        .key:hover { background: #555; }
        #done { display: none; font-size: 28px; margin-top: 80px; color: #0f0; }
        #skiplabel { color: #888; margin-top: 10px; font-size: 13px; }
    </style>
</head>
<body>
    <div id="top">
        <b>Bus Color Sorter</b>
        <span id="counts">
            <span class="c-red">red: <span id="cnt-red">0</span></span>
            <span class="c-blue">blue: <span id="cnt-blue">0</span></span>
            <span class="c-yellow">yellow: <span id="cnt-yellow">0</span></span>
            <span class="c-rest">rest: <span id="cnt-rest">0</span></span>
        </span>
    </div>
    <div id="imgwrap">
        <img id="img" src="">
        <div id="name"></div>
        <div id="remaining"></div>
    </div>
    <div id="keys">
        <span class="key c-red" onclick="classify('red')">1 = Red</span>
        <span class="key c-blue" onclick="classify('blue')">2 = Blue</span>
        <span class="key c-yellow" onclick="classify('yellow')">3 = Yellow</span>
        <span class="key c-rest" onclick="classify('rest')">4 = Rest</span>
        <span class="key" onclick="undo()">Backspace = Undo</span>
        <span class="key" onclick="skip()">S = Skip</span>
    </div>
    <div id="skiplabel">Skip leaves the crop unsorted in the crops dir; it comes back at the end of the queue.</div>
    <div id="done">All crops sorted.</div>

<script>
let current = null;

function refreshCounts(counts) {
    for (const c of ["red", "blue", "yellow", "rest"]) {
        document.getElementById("cnt-" + c).textContent = counts[c] || 0;
    }
}

async function loadNext() {
    const res = await fetch("/api/next");
    const data = await res.json();
    refreshCounts(data.counts);
    if (data.done) {
        document.getElementById("imgwrap").style.display = "none";
        document.getElementById("keys").style.display = "none";
        document.getElementById("done").style.display = "block";
        current = null;
        return;
    }
    current = data.name;
    document.getElementById("img").src = "/img/" + encodeURIComponent(data.name);
    document.getElementById("name").textContent = data.name;
    document.getElementById("remaining").textContent = data.remaining + " remaining";
}

async function classify(color) {
    if (!current) return;
    const res = await fetch("/api/classify", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({name: current, color: color}),
    });
    const data = await res.json();
    if (!data.ok) { alert(data.error || "classify failed"); return; }
    loadNext();
}

async function skip() {
    if (!current) return;
    await fetch("/api/skip", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({name: current}),
    });
    loadNext();
}

async function undo() {
    const res = await fetch("/api/undo", {method: "POST"});
    const data = await res.json();
    if (!data.ok) { alert(data.error || "nothing to undo"); return; }
    loadNext();
}

window.addEventListener("keydown", (e) => {
    if (e.key === "1") classify("red");
    else if (e.key === "2") classify("blue");
    else if (e.key === "3") classify("yellow");
    else if (e.key === "4") classify("rest");
    else if (e.key === "Backspace") { e.preventDefault(); undo(); }
    else if (e.key === "s" || e.key === "S") skip();
});

loadNext();
</script>
</body>
</html>
"""


class SorterState:
    def __init__(self, crops_dir, dataset_dir):
        self.crops_dir = crops_dir
        self.train_dir = dataset_dir / "train"
        for c in COLORS:
            (self.train_dir / c).mkdir(parents=True, exist_ok=True)
        self.skipped = set()   # names skipped this session; requeued after non-skipped ones
        self.last_move = None  # (src_path, dst_path) for undo

    def _queue(self):
        names = sorted(p.name for p in self.crops_dir.iterdir() if p.is_file())
        not_skipped = [n for n in names if n not in self.skipped]
        skipped_first = [n for n in names if n in self.skipped]
        return not_skipped + skipped_first

    def counts(self):
        return {c: len(list((self.train_dir / c).glob("*"))) for c in COLORS}

    def next_item(self):
        queue = self._queue()
        if not queue:
            return None, 0
        return queue[0], len(queue)

    def classify(self, name, color):
        if color not in COLORS:
            raise ValueError(f"invalid color: {color}")
        src = self.crops_dir / name
        if not src.is_file():
            raise FileNotFoundError(name)
        dst = self.train_dir / color / name
        shutil.move(str(src), str(dst))
        self.skipped.discard(name)
        self.last_move = (src, dst)

    def skip(self, name):
        self.skipped.add(name)

    def undo(self):
        if self.last_move is None:
            raise RuntimeError("nothing to undo")
        src, dst = self.last_move
        if not dst.is_file():
            raise RuntimeError("last classified file no longer where expected; can't undo")
        shutil.move(str(dst), str(src))
        self.last_move = None


def make_handler(state: SorterState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                body = HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/next":
                name, remaining = state.next_item()
                if name is None:
                    self._json({"done": True, "counts": state.counts(), "remaining": 0})
                else:
                    self._json({"done": False, "name": name, "remaining": remaining, "counts": state.counts()})
            elif self.path.startswith("/img/"):
                from urllib.parse import unquote
                name = unquote(self.path[len("/img/"):])
                path = state.crops_dir / name
                if ".." in name or not path.is_file():
                    self.send_response(404)
                    self.end_headers()
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {}

            if self.path == "/api/classify":
                try:
                    state.classify(payload["name"], payload["color"])
                    self._json({"ok": True})
                except Exception as e:
                    self._json({"ok": False, "error": str(e)}, 400)
            elif self.path == "/api/skip":
                state.skip(payload.get("name", ""))
                self._json({"ok": True})
            elif self.path == "/api/undo":
                try:
                    state.undo()
                    self._json({"ok": True})
                except Exception as e:
                    self._json({"ok": False, "error": str(e)}, 400)
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops-dir", type=Path, required=True, help="Output dir from `run.py collect_crops`.")
    p.add_argument("--dataset-dir", type=Path, required=True,
                    help="Where to write train/red,blue,yellow,rest (same dir you'll pass to "
                         "`run.py train_classifier --color-dataset-dir`).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.crops_dir.is_dir():
        raise SystemExit(f"--crops-dir not found: {args.crops_dir}")

    state = SorterState(args.crops_dir, args.dataset_dir)
    hostname = socket.gethostname()
    print(f"Crops dir:   {args.crops_dir}")
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"On your Mac:\n  ssh -L {args.port}:localhost:{args.port} oauser@{hostname}\n  open http://localhost:{args.port}")
    print("\nKeys: 1=red 2=blue 3=yellow 4=rest | Backspace=undo | S=skip")

    server = HTTPServer((args.host, args.port), make_handler(state))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
