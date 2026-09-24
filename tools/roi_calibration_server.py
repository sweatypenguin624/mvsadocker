import os
import cv2
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn
from pathlib import Path

app = FastAPI()

CONFIG_PATH = "/home/users/oauser/mvsa/vehicle-counting/config/vehicle_count_config.yaml"

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>ROI Calibration</title>
    <style>
        body { font-family: sans-serif; padding: 20px; background-color: #121212; color: #ffffff; }
        .controls { margin-bottom: 20px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        #canvas-container { position: relative; display: inline-block; box-shadow: 0 4px 8px rgba(0,0,0,0.5); }
        canvas { position: absolute; top: 0; left: 0; cursor: crosshair; }
        img { display: block; max-width: 100%; height: auto; border-radius: 4px; }
        #coordinates { margin-top: 15px; font-family: monospace; font-size: 16px; color: #4caf50; }
        .btn { padding: 10px 20px; cursor: pointer; background-color: #2196f3; color: white; border: none; border-radius: 4px; transition: background 0.3s; }
        .btn:hover { background-color: #1976d2; }
        .btn-success { background-color: #4caf50; }
        .btn-success:hover { background-color: #388e3c; }
        .btn-danger { background-color: #f44336; }
        .btn-danger:hover { background-color: #d32f2f; }
        input[type="text"] { padding: 10px; border-radius: 4px; border: 1px solid #555; background: #222; color: #fff; width: 400px;}
    </style>
</head>
<body>
    <h2>ROI Calibration Tool</h2>
    <div class="controls">
        <label>Video Path:</label>
        <input type="text" id="video-path" value="/home/users/oauser/mvsa/realrun/test.mp4">
        <button class="btn" onclick="loadFrame()">Load Frame</button>
        <button class="btn btn-danger" onclick="resetLine()">Reset Line</button>
        <button class="btn btn-success" onclick="saveConfig()">Save to Config</button>
    </div>
    
    <div id="canvas-container">
        <img id="video-frame" src="" alt="Video Frame" style="display:none;" onload="setupCanvas()">
        <canvas id="overlay"></canvas>
    </div>
    
    <div id="coordinates">Points: None</div>
    
    <script>
        let points = [];
        let originalWidth = 0;
        let originalHeight = 0;
        let img = document.getElementById('video-frame');
        let canvas = document.getElementById('overlay');
        let ctx = canvas.getContext('2d');
        
        function loadFrame() {
            const path = document.getElementById('video-path').value;
            if (!path) return alert("Enter video path");
            img.src = "/frame?video_path=" + encodeURIComponent(path);
            img.style.display = "block";
            points = [];
            updateCoordinatesText();
        }
        
        function setupCanvas() {
            canvas.width = img.width;
            canvas.height = img.height;
            
            fetch('/dimensions?video_path=' + encodeURIComponent(document.getElementById('video-path').value))
                .then(r => r.json())
                .then(data => {
                    originalWidth = data.width;
                    originalHeight = data.height;
                });
                
            draw();
        }
        
        canvas.addEventListener('mousedown', function(e) {
            if (points.length >= 2) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const scaleX = originalWidth / canvas.width;
            const scaleY = originalHeight / canvas.height;
            
            const origX = Math.round(x * scaleX);
            const origY = Math.round(y * scaleY);
            
            points.push([origX, origY]);
            draw();
            updateCoordinatesText();
        });
        
        function resetLine() {
            points = [];
            draw();
            updateCoordinatesText();
        }
        
        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (points.length > 0) {
                const scaleX = canvas.width / originalWidth;
                const scaleY = canvas.height / originalHeight;
                
                ctx.fillStyle = "#00ff00";
                for (let i = 0; i < points.length; i++) {
                    const px = points[i][0] * scaleX;
                    const py = points[i][1] * scaleY;
                    ctx.beginPath();
                    ctx.arc(px, py, 6, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.strokeStyle = "white";
                    ctx.stroke();
                }
                
                if (points.length == 2) {
                    ctx.strokeStyle = "#ff0000";
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.moveTo(points[0][0] * scaleX, points[0][1] * scaleY);
                    ctx.lineTo(points[1][0] * scaleX, points[1][1] * scaleY);
                    ctx.stroke();
                }
            }
        }
        
        function updateCoordinatesText() {
            document.getElementById('coordinates').innerText = "Points (original scale): " + JSON.stringify(points);
        }
        
        function saveConfig() {
            if (points.length !== 2) {
                alert("Please draw exactly 2 points.");
                return;
            }
            fetch('/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({points: points})
            }).then(r => r.json()).then(data => {
                if (data.status === "success") {
                    alert("Saved coordinates to config successfully!");
                } else {
                    alert("Error: " + data.message);
                }
            });
        }
    </script>
</body>
</html>
"""

@app.get("/")
def index():
    return HTMLResponse(content=HTML_CONTENT)

@app.get("/frame")
def get_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return Response(content="Could not open video", status_code=400)
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return Response(content="Could not read frame", status_code=400)
    
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        return Response(content="Failed to encode image", status_code=500)
    
    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")

@app.get("/dimensions")
def get_dimensions(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return JSONResponse({"error": "could not open"}, status_code=400)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"width": width, "height": height}

class SaveRequest(BaseModel):
    points: list

@app.post("/save")
def save_config(req: SaveRequest):
    if len(req.points) != 2:
        return JSONResponse({"status": "error", "message": "Need exactly 2 points"})
        
    try:
        with open(CONFIG_PATH, "r") as f:
            lines = f.readlines()
            
        out_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            out_lines.append(line)
            if line.strip().startswith("counting_line:"):
                out_lines.append(f"    - - {req.points[0][0]}\n")
                out_lines.append(f"      - {req.points[0][1]}\n")
                out_lines.append(f"    - - {req.points[1][0]}\n")
                out_lines.append(f"      - {req.points[1][1]}\n")
                
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.strip() == "" or next_line.lstrip() == next_line or next_line.strip().startswith("count_direction:") or next_line.strip().startswith("road_polygon:"):
                        break
                    i += 1
                continue
            i += 1
            
        with open(CONFIG_PATH, "w") as f:
            f.writelines(out_lines)
            
        return {"status": "success"}
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

if __name__ == "__main__":
    print(f"Starting ROI Calibration Web Server...")
    print(f"Open http://localhost:8000 in your browser.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
