"""YOLO26-experiment counting-line editor -- draw a 2-point counting line on a
still frame and save it straight into this experiment's own
stage1_config_yolo26.yaml (never the shared config/stage1_config.yaml).

Modelled on scripts/roieditor.py's browser-canvas pattern, but for exactly
one line (the counting line Stage 1's tracker/counter reads), not a
polygon.

Usage (one camera at a time):

    env/bin/python scripts/traffic_stage1_yolo26/line_editor.py \\
        --camera-key tp00075 \\
        --image scripts/traffic_stage1_yolo26/frames/tp00075_frame300.jpg

Then, on your Mac:
    ssh -L 8766:localhost:8766 oauser@<host>
    open http://localhost:8766

Left click = place point (keeps only the last 2) | R = reset | S = save
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

HTML = r"""
<!DOCTYPE html>
<html>
<head>
    <title>YOLO26 Line Editor -- __CAMERA_KEY__</title>
    <style>
        body { margin: 0; background: #111; color: white; font-family: Arial, sans-serif; }
        #top { padding: 12px 18px; background: #222; }
        #container { position: relative; display: inline-block; margin: 20px; }
        #image { display: block; max-width: 90vw; max-height: 80vh; }
        #canvas { position: absolute; left: 0; top: 0; cursor: crosshair; }
        button { margin-right: 8px; padding: 8px 14px; cursor: pointer; }
        #coords { margin-top: 10px; white-space: pre; font-family: monospace; color: #0f0; }
        #dims { color: #ff0; }
    </style>
</head>
<body>
<div id="top">
    <b>YOLO26 Line Editor -- camera: __CAMERA_KEY__</b>
    <br><br>
    Left click = place a point (only the last 2 are kept) | R = reset | S = save
    <br><br>
    <button onclick="resetPoints()">Reset</button>
    <button onclick="saveLine()">Save Line</button>
    <div id="dims"></div>
    <div id="coords"></div>
</div>
<div id="container">
    <img id="image" src="/image">
    <canvas id="canvas"></canvas>
</div>
<script>
const image = document.getElementById("image");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let points = [];
let naturalWidth = 0;
let naturalHeight = 0;

image.onload = function() {
    naturalWidth = image.naturalWidth;
    naturalHeight = image.naturalHeight;
    document.getElementById("dims").textContent =
        "Image native resolution: " + naturalWidth + " x " + naturalHeight +
        " (saved as reference_width/reference_height)";
    canvas.width = image.clientWidth;
    canvas.height = image.clientHeight;
    redraw();
};

function getScale() {
    return { x: image.clientWidth / naturalWidth, y: image.clientHeight / naturalHeight };
}

function getOriginalCoordinates(event) {
    const rect = canvas.getBoundingClientRect();
    const displayX = event.clientX - rect.left;
    const displayY = event.clientY - rect.top;
    const scale = getScale();
    return { x: Math.round(displayX / scale.x), y: Math.round(displayY / scale.y) };
}

canvas.addEventListener("click", function(event) {
    points.push(getOriginalCoordinates(event));
    if (points.length > 2) points.shift();
    redraw();
});

function redraw() {
    canvas.width = image.clientWidth;
    canvas.height = image.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (points.length === 2) {
        const scale = getScale();
        const [a, b] = points;
        ctx.beginPath();
        ctx.moveTo(a.x * scale.x, a.y * scale.y);
        ctx.lineTo(b.x * scale.x, b.y * scale.y);
        ctx.strokeStyle = "yellow";
        ctx.lineWidth = 3;
        ctx.stroke();
    }
    points.forEach((p, i) => {
        const scale = getScale();
        const x = p.x * scale.x, y = p.y * scale.y;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = "red";
        ctx.fill();
        ctx.fillStyle = "white";
        ctx.font = "14px Arial";
        ctx.fillText(`${i}: (${p.x}, ${p.y})`, x + 10, y - 10);
    });
    updateText();
}

function updateText() {
    let text = "line endpoints (native pixels):\n";
    points.forEach(p => { text += `    [${p.x}, ${p.y}],\n`; });
    document.getElementById("coords").textContent = text;
}

function resetPoints() { points = []; redraw(); }

function saveLine() {
    if (points.length !== 2) { alert("Place exactly 2 points first (click twice)."); return; }
    fetch("/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: points, width: naturalWidth, height: naturalHeight })
    })
    .then(response => response.text())
    .then(text => alert(text));
}

document.addEventListener("keydown", function(event) {
    if (event.key === "r" || event.key === "R") resetPoints();
    if (event.key === "s" || event.key === "S") saveLine();
});
</script>
</body>
</html>
"""


def set_line(config_path: Path, camera_key: str, line, ref_w: int, ref_h: int) -> None:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cameras = data.setdefault("cameras", {})
    if cameras is None:
        cameras = data["cameras"] = {}
    block = cameras.setdefault(camera_key, {})
    if block is None:
        block = cameras[camera_key] = {}
    counting = block.setdefault("counting", {})
    if counting is None:
        counting = block["counting"] = {}
    counting["line"] = [[line[0], line[1]], [line[2], line[3]]]
    counting["reference_width"] = ref_w
    counting["reference_height"] = ref_h
    cameras[camera_key] = block
    data["cameras"] = cameras
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def make_handler(image_path: Path, camera_key: str, config_path: Path):
    html = HTML.replace("__CAMERA_KEY__", camera_key)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                body = html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/image":
                if not image_path.exists():
                    self.send_error(404, "Image not found")
                    return
                data = image_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)

        def do_POST(self):
            if self.path != "/save":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode())
            points = payload["points"]
            width = int(payload["width"])
            height = int(payload["height"])
            if len(points) != 2:
                self.send_error(400, "Need exactly 2 points")
                return

            line = [points[0]["x"], points[0]["y"], points[1]["x"], points[1]["y"]]
            set_line(config_path, camera_key, line, width, height)

            msg = (
                f"Saved counting line for '{camera_key}' ({width}x{height}) -> "
                f"{config_path}: {line}"
            )
            print(msg)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(msg.encode())

        def log_message(self, format, *args):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-key", required=True, help="e.g. tp00075")
    parser.add_argument("--image", type=Path, required=True, help="Native-resolution still frame for this camera")
    parser.add_argument(
        "--config", type=Path,
        default=HERE / "stage1_config_yolo26.yaml",
        help="YOLO26 experiment config to write the line into (defaults to this folder's own config).",
    )
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"image not found: {args.image}")
    if not args.config.exists():
        raise FileNotFoundError(f"config not found: {args.config}")

    print()
    print("YOLO26 Line Editor")
    print("==============================")
    print(f"Camera key : {args.camera_key}")
    print(f"Image      : {args.image}")
    print(f"Config     : {args.config}")
    print()
    print(f"Listening on port {args.port}")
    print()
    print("On your Mac, create an SSH tunnel to THIS SAME HOST:")
    print()
    print(f"ssh -L {args.port}:localhost:{args.port} {os.environ.get('USER', 'oauser')}@{socket.getfqdn()}")
    print()
    print("Then open:")
    print()
    print(f"http://localhost:{args.port}")
    print()

    handler = make_handler(args.image, args.camera_key, args.config)
    server = HTTPServer(("127.0.0.1", args.port), handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
