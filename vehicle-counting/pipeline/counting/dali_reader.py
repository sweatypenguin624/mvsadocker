"""
DALI Video Stream Reader with NVDEC Hardware Acceleration.
Located entirely within vehicle-counting/pipeline/counting/.
"""
from pathlib import Path
from typing import Iterator, Tuple
import numpy as np
import cv2

try:
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    HAS_DALI = True
except ImportError:
    HAS_DALI = False


class DALIVideoStreamReader:
    """Streams video frames using NVIDIA DALI and GPU NVDEC hardware decoding."""

    def __init__(self, video_path: Path, device_id: int = 0):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        # Probe metadata via OpenCV
        cap = cv2.VideoCapture(str(self.video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.device_id = device_id
        self.pipe = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def frames(self, start_frame: int = 0, keep=None) -> Iterator[Tuple[int, np.ndarray]]:
        if not HAS_DALI:
            raise RuntimeError("NVIDIA DALI is not installed in the environment.")

        @pipeline_def
        def video_pipe(filenames):
            return fn.readers.video(
                device="gpu",
                filenames=filenames,
                sequence_length=1,
                initial_fill=1,
                random_shuffle=False,
                shard_id=0,
                num_shards=1
            )
        
        self.pipe = video_pipe(
            filenames=[str(self.video_path)],
            batch_size=1,
            num_threads=2,
            device_id=self.device_id
        )
        self.pipe.build()

        frame_idx = 0
        while True:
            try:
                out = self.pipe.run()
                if frame_idx < start_frame or (keep is not None and not keep(frame_idx)):
                    frame_idx += 1
                    continue
                # out[0] is TensorListGPU, as_cpu().as_array() gives (1, 1, H, W, 3) in RGB
                frame_rgb = out[0].as_cpu().as_array()[0, 0]
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                yield frame_idx, frame_bgr
                frame_idx += 1
            except StopIteration:
                break
            except Exception:
                break

    def close(self):
        self.pipe = None


# python vehicle-counting/pipeline/counting/main.py \
#     --video "/home/users/oauser/mvsa/realrun/test.mp4" \
#     --output_dir "/home/users/oauser/mvsa/realrun/output" \
#     --no_annotation \
#     --use_dali
