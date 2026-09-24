import os
import re
import json
import yaml
import socket
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path("/home/users/oauser/mvsa/vehicle-count-output")
PORT = 8765

os.chdir(ROOT)

CAM_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
SAVE_LOCK = threading.Lock()

def atomic_write_yaml(path: Path, data):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    os.replace(tmp_path, path)

class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if self.path == "/save_line":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                cam = data.get("camera")
                points = data.get("points")

                if not cam or not CAM_NAME_RE.match(cam):
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": "Invalid or missing camera name"}).encode())
                    return

                if (not points or len(points) != 2
                        or any(len(p) != 2 for p in points)
                        or any(not isinstance(v, (int, float)) for p in points for v in p)):
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": "Expected 2 numeric [x, y] points"}).encode())
                    return

                points = [[int(round(v)) for v in p] for p in points]

                base_config_path = Path("/home/users/oauser/mvsa/vehicle-counting/config/vehicle_count_config.yaml")
                cam_cfg_path = Path(f"/home/users/oauser/mvsa/vehicle-counting/config/vehicle_count_config_{cam}.yaml")

                # Serialize concurrent saves so no two requests read/modify/write
                # the base template at the same time (ThreadingHTTPServer runs
                # each request on its own thread).
                with SAVE_LOCK:
                    # Each camera starts from the shared base template, but only
                    # the camera-specific file is ever written back — saving one
                    # camera must never change what another camera's line was.
                    with open(base_config_path, "r") as f:
                        cfg = yaml.safe_load(f)

                    cfg.setdefault("roi", {})
                    cfg["roi"]["counting_line"] = points

                    atomic_write_yaml(cam_cfg_path, cfg)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success",
                    "saved_file": str(cam_cfg_path),
                    "points": points
                }).encode())
                print(f"[Line Saved] Camera {cam}: {points} -> {cam_cfg_path}")
                return
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode())
                return
        else:
            self.send_error(404)

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            pass

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()
            
        if "Range" not in self.headers:
            self.range_length = None
            return super().send_head()

        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        fs = os.fstat(f.fileno())
        file_len = fs.st_size
        ctype = self.guess_type(path)

        range_header = self.headers.get("Range", "").strip()
        match = re.match(r"^bytes=(\d*)-(\d*)$", range_header)
        if not match:
            f.close()
            self.range_length = None
            return super().send_head()

        start_str, end_str = match.groups()

        if not start_str and not end_str:
            f.close()
            self.send_error(400, "Bad Request")
            return None

        if start_str:
            start = int(start_str)
            if end_str:
                end = int(end_str)
            else:
                end = file_len - 1
        else:
            start = file_len - int(end_str)
            end = file_len - 1

        if start < 0:
            start = 0

        if start >= file_len or end >= file_len or start > end:
            f.close()
            self.send_response(416, "Requested Range Not Satisfiable")
            self.send_header("Content-Range", f"bytes */{file_len}")
            self.send_header("Content-Length", "0")
            self.send_header("Content-Type", ctype)
            self.end_headers()
            return None

        chunk_len = end - start + 1
        
        self.send_response(206, "Partial Content")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(chunk_len))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_len}")
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        
        f.seek(start)
        self.range_length = chunk_len
        return f

    def copyfile(self, source, outputfile):
        if not getattr(self, "range_length", None):
            super().copyfile(source, outputfile)
            return
            
        remaining = self.range_length
        chunk_size = 1024 * 1024 * 4  # 4 MB chunks
        
        while remaining > 0:
            read_size = min(remaining, chunk_size)
            data = source.read(read_size)
            if not data:
                break
            
            try:
                outputfile.write(data)
            except (ConnectionResetError, BrokenPipeError):
                break
                
            remaining -= len(data)

ThreadingHTTPServer.allow_reuse_address = True
server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)

if __name__ == "__main__":
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "SERVER_IP"

    print("==================================================")
    print(f" Serving directory : {ROOT}")
    print(f" Listening on port : {PORT}")
    print("--------------------------------------------------")
    print(f" Local Access      : http://localhost:{PORT}/")
    print(f" Remote/LAN Access : http://{local_ip}:{PORT}/")
    print(f" SSH Tunnel Cmd    : ssh -L {PORT}:localhost:{PORT} user@{hostname}")
    print("==================================================")
    print("Press Ctrl+C to stop.")
    server.serve_forever()
