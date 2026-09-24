"""MVSA ROI editor -- draw a pedestrian ROI polygon on a still frame and save
it into the shared per-camera ROI config (config/roi/roi_by_camera.yaml).

Rewritten from the original single-image/single-output version to fix the
exact bug that produced a bad polygon previously: the saved output now
always includes the image's ACTUAL pixel dimensions (image.naturalWidth/
naturalHeight, read straight from the loaded <img> in the browser) as
reference_width/reference_height, so the polygon can never silently end up
scaled against the wrong reference later -- that mismatch (drawing on one
resolution, applying against another) is what caused points to land outside
the visible frame at runtime.

Usage (one camera at a time -- pick the next --camera-key, run, draw, save,
Ctrl+C, repeat for the next camera):

    python scripts/roieditor.py --camera-key peds_15_koppikkar_road__cam_2 \\
        --image config/roi/frames/peds_15_koppikkar_road__cam_2.jpg

Then, on your Mac:
    ssh -L 8765:localhost:8765 oauser@<host>
    open http://localhost:8765

Left click = add point | Right click = remove last point | R = reset | S = save
Saving upserts this camera's entry into config/roi/roi_by_camera.yaml
(existing entries for other cameras are preserved).
"""

import argparse
import json
import os
import socket
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Deliberately stdlib-only (no PyYAML) -- this script is run directly with
# whatever bare "python3" is local to whichever node you're SSH'd into
# (which may not be the ~/mvsa/env venv -- see the reference_width/height
# mismatch note above), and pip-installing anything into that node's
# system/user site-packages would violate the "everything lives under
# ~/mvsa" project requirement. The saved file is plain JSON, which is a
# strict YAML subset, so pedestrian_counter.py's yaml.safe_load() (running
# under the venv, which does have PyYAML) reads it back with no changes.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

HTML = r"""
<!DOCTYPE html>
<html>
<head>
    <title>MVSA ROI Editor -- __CAMERA_KEY__</title>
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
    <b>MVSA ROI Editor -- camera: __CAMERA_KEY__</b>
    <br><br>
    Left click = add point | Right click = remove last point | R = reset | S = save
    <br><br>
    <button onclick="resetPoints()">Reset</button>
    <button onclick="saveROI()">Save ROI</button>
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
        " (this is what will be saved as reference_width/reference_height)";
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
    redraw();
});

canvas.addEventListener("contextmenu", function(event) {
    event.preventDefault();
    if (points.length > 0) { points.pop(); redraw(); }
});

function redraw() {
    canvas.width = image.clientWidth;
    canvas.height = image.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (points.length === 0) { updateText(); return; }
    const scale = getScale();
    ctx.beginPath();
    points.forEach((p, i) => {
        const x = p.x * scale.x, y = p.y * scale.y;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    if (points.length >= 3) {
        ctx.closePath();
        ctx.fillStyle = "rgba(255, 255, 0, 0.20)";
        ctx.fill();
    }
    ctx.strokeStyle = "yellow";
    ctx.lineWidth = 3;
    ctx.stroke();
    points.forEach((p, i) => {
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
    let text = "polygon (native pixels):\n";
    points.forEach(p => { text += `    [${p.x}, ${p.y}],\n`; });
    document.getElementById("coords").textContent = text;
}

function resetPoints() { points = []; redraw(); }

function saveROI() {
    if (points.length < 3) { alert("Add at least 3 points first."); return; }
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
    if (event.key === "s" || event.key === "S") saveROI();
});
</script>
</body>
</html>
"""


def make_handler(image_path: Path, camera_key: str, roi_config_path: Path):
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

            roi_config_path.parent.mkdir(parents=True, exist_ok=True)
            all_rois = {}
            if roi_config_path.exists():
                with open(roi_config_path, "r", encoding="utf-8") as f:
                    all_rois = json.load(f) or {}

            all_rois[camera_key] = {
                "polygon": [[int(p["x"]), int(p["y"])] for p in points],
                "reference_width": width,
                "reference_height": height,
                "source_image": str(image_path),
                "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

            # Plain JSON -- a strict subset of YAML, so pedestrian_counter.py's
            # yaml.safe_load() (under the venv, where PyYAML is available)
            # reads this back unchanged. Deliberately not using PyYAML here
            # (see module docstring).
            with open(roi_config_path, "w", encoding="utf-8") as f:
                json.dump(all_rois, f, indent=2, sort_keys=True)

            msg = f"Saved {len(points)}-point ROI for '{camera_key}' ({width}x{height}) -> {roi_config_path}"
            print(msg)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(msg.encode())

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-key", required=True, help="e.g. peds_15_koppikkar_road__cam_2")
    parser.add_argument("--image", type=Path, required=True, help="Native-resolution still frame for this camera")
    parser.add_argument("--roi-config", type=Path, default=PROJECT_ROOT / "config" / "roi" / "roi_by_camera.yaml")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"image not found: {args.image}")

    print()
    print("MVSA ROI Editor")
    print("==============================")
    print(f"Camera key : {args.camera_key}")
    print(f"Image      : {args.image}")
    print(f"ROI config : {args.roi_config}")
    print()
    print(f"Listening on port {args.port}")
    print()
    print("On your Mac, create an SSH tunnel to THIS SAME HOST (the tunnel target")
    print("must match where this server is actually running, not a login alias")
    print("that a cluster jump host might route you to a different node):")
    print()
    print(f"ssh -L {args.port}:localhost:{args.port} {os.environ.get('USER', 'oauser')}@{socket.getfqdn()}")
    print()
    print("Then open:")
    print()
    print(f"http://localhost:{args.port}")
    print()

    handler = make_handler(args.image, args.camera_key, args.roi_config)
    server = HTTPServer(("127.0.0.1", args.port), handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
