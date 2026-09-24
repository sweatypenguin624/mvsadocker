
import cv2
import json
import csv
from pathlib import Path
import subprocess
import threading
import queue
try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


class FFmpegStreamer:
    def __init__(self, fps, width, height, port=9999):
        if imageio_ffmpeg is None:
            raise RuntimeError("imageio-ffmpeg is not installed. Run: pip install imageio-ffmpeg")
        self.ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        
        command = [
            self.ffmpeg_path,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{int(width)}x{int(height)}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-f", "mpegts",
            f"tcp://0.0.0.0:{port}?listen=1"
        ]
        
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.q = queue.Queue(maxsize=30)
        self.running = True
        
        self.thread = threading.Thread(target=self._writer_thread)
        self.thread.daemon = True
        self.thread.start()
        
    def _writer_thread(self):
        while self.running:
            try:
                frame = self.q.get(timeout=0.1)
                if frame is None:
                    break
                self.process.stdin.write(frame.tobytes())
            except queue.Empty:
                continue
            except Exception as e:
                break
                
    def write_frame(self, frame):
        if not self.running: return
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            pass
            
    def close(self):
        self.running = False
        try:
            self.q.put(None, timeout=1.0)
        except:
            pass
        self.thread.join(timeout=2.0)
        try:
            self.process.terminate()
        except:
            pass

class Annotator:

    def __init__(self, output_path, fps, width, height, debug_mode=False, live_stream=False):
        self.output_path = output_path
        ensure_dir(Path(output_path).parent)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (int(width), int(height)))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter at {self.output_path}")
        print(f"VideoWriter initialized at: {Path(self.output_path).resolve()}")
        self.debug_mode = debug_mode
        self.streamer = None
        if live_stream:
            try:
                self.streamer = FFmpegStreamer(fps, width, height, port=9999)
                print("Live stream enabled. Connect via: ffplay tcp://localhost:9999")
            except Exception as e:
                print(f"Failed to start FFmpegStreamer: {e}")
        
    def draw_line(self, frame, pt1, pt2):
        cv2.line(frame, tuple(map(int, pt1)), tuple(map(int, pt2)), (0, 0, 255), 3)

    def draw_track(self, frame, box, tid, cls_name, conf, counted, state_str="NEW", not_counted_reason=""):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 255) if counted and not_counted_reason == "Duplicate" else ((0, 255, 0) if counted else (255, 0, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        if self.debug_mode:
            label = f"ID:{tid} | {cls_name} | {conf:.2f} | {state_str}"
        else:
            label = cls_name
        
        cv2.putText(frame, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    def write_frame(self, frame):
        self.writer.write(frame)
        if self.streamer:
            self.streamer.write_frame(frame)
        
    def close(self):
        self.writer.release()
        if self.streamer:
            self.streamer.close()

def export_csv(records, path):
    ensure_dir(Path(path).parent)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_or_frame", "class", "count", "direction", "track_id"])
        for r in records:
            writer.writerow([r["frame"], r["class"], 1, r["direction"], r["track_id"]])

def export_json(summary, path):
    ensure_dir(Path(path).parent)
    with open(path, 'w') as f:
        json.dump(summary, f, indent=4)
