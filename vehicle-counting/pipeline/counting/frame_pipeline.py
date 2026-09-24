"""Overlaps video decode, YOLO detection and tracking without changing results.

Detection of frame N+1 does not depend on tracking frame N, so decoding and
``model.predict`` run in background threads while the caller applies the
tracker strictly in frame order, with exactly the inputs ``model.track()``
would have fed it (see ultralytics/trackers/track.py:on_predict_postprocess_end).
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Iterator, Optional

import numpy as np

_END = object()
_POLL = 0.1


class FrameSampler:
    """The processing_fps frame-selection rule main.py has always used."""

    def __init__(self, fps: float, processing_fps: float, start_frame: int):
        self.fps = fps
        self.start_frame = start_frame
        self.processing_fps = processing_fps
        self.time_per_frame = 1.0 / processing_fps if processing_fps > 0 else 0
        self.next_process_time = 0.0

    def keep(self, frame_idx: int) -> bool:
        frame_time = (frame_idx - self.start_frame) / self.fps
        if self.processing_fps > 0 and frame_time < self.next_process_time - 1e-5:
            return False
        self.next_process_time += self.time_per_frame
        return True


# Tracker types whose update() needs nothing beyond (boxes, img); others need
# predictor hooks (ReID features, frame extras) and must run through model.track().
PIPELINE_SAFE_TRACKERS = {"bytetrack", "botsort"}


def build_tracker(tracker_yaml: str, device):
    """Build the tracker exactly as ultralytics on_predict_start does. Returns
    None if this tracker config cannot be driven outside model.track()."""
    try:
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml
    except ImportError:
        return None
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_yaml)))
    cfg.device = device
    if cfg.tracker_type not in PIPELINE_SAFE_TRACKERS or getattr(cfg, "with_reid", False):
        return None
    tracker_cls = TRACKER_MAP[cfg.tracker_type]
    if hasattr(tracker_cls, "setup_predictor") or hasattr(tracker_cls, "compute_frame_extras"):
        return None
    return tracker_cls(args=cfg)


def apply_tracker(tracker, result):
    """Mirror of ultralytics on_predict_postprocess_end for one non-OBB result."""
    import torch

    src = result.boxes
    det = src.cpu().numpy()
    tracks = tracker.update(det, result.orig_img, feats=getattr(result, "feats", None))
    if len(tracks) == 0:
        return result
    idx = tracks[:, -1].astype(int)
    result = result[idx]
    result.update(boxes=torch.as_tensor(tracks[:, :-1], device=src.data.device))
    return result


class FramePipeline:
    """Yields ``(frame_idx, frame, result)`` in frame order.

    - A decode thread pulls kept frames from ``frame_iter`` (bounded by ``max_inflight``).
    - With ``predict_fns`` (one per worker, each owning its own model instance),
      worker threads run detection and ``result`` is the untracked prediction.
    - Without ``predict_fns`` it is a pure decode prefetcher and ``result`` is None.
    """

    def __init__(self, frame_iter: Iterator, max_inflight: int = 16,
                 predict_fns: Optional[list[Callable[[np.ndarray], object]]] = None):
        self._frame_iter = frame_iter
        self._predict_fns = predict_fns or []
        self._stop = threading.Event()
        self._slots = threading.Semaphore(max(2, max_inflight))
        self._decoded: queue.Queue = queue.Queue()
        self._done: dict[int, tuple] = {}
        self._done_cv = threading.Condition()
        self._total: Optional[int] = None
        self._error: Optional[BaseException] = None

        self.decode_busy_s = 0.0
        self.detect_busy_s = 0.0
        self.yolo_speed_s = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
        self.wait_s = 0.0
        self.frames_yielded = 0
        self.frames_detected = 0

        self._threads = [threading.Thread(target=self._decode_loop, name="decode", daemon=True)]
        for i, fn in enumerate(self._predict_fns):
            self._threads.append(threading.Thread(target=self._detect_loop, args=(fn,),
                                                  name=f"detect-{i}", daemon=True))
        self._stats_lock = threading.Lock()
        for t in self._threads:
            t.start()

    def _fail(self, exc: BaseException) -> None:
        with self._done_cv:
            if self._error is None:
                self._error = exc
            self._done_cv.notify_all()
        self._stop.set()

    def _acquire_slot(self) -> bool:
        while not self._stop.is_set():
            if self._slots.acquire(timeout=_POLL):
                return True
        return False

    def _decode_loop(self) -> None:
        seq = 0
        try:
            it = iter(self._frame_iter)
            while self._acquire_slot():
                t0 = time.perf_counter()
                item = next(it, _END)
                self.decode_busy_s += time.perf_counter() - t0
                if item is _END:
                    self._slots.release()
                    break
                if self._predict_fns:
                    self._decoded.put((seq, item))
                else:
                    self._publish(seq, (item[0], item[1], None))
                seq += 1
        except BaseException as e:
            self._fail(e)
        finally:
            with self._done_cv:
                self._total = seq
                self._done_cv.notify_all()
            for _ in self._predict_fns:
                self._decoded.put(_END)
            close = getattr(self._frame_iter, "close", None)
            if close:
                close()

    def _detect_loop(self, predict_fn) -> None:
        try:
            while not self._stop.is_set():
                try:
                    job = self._decoded.get(timeout=_POLL)
                except queue.Empty:
                    continue
                if job is _END:
                    return
                seq, (frame_idx, frame) = job
                t0 = time.perf_counter()
                result = predict_fn(frame)
                busy = time.perf_counter() - t0
                speed = getattr(result, "speed", None) or {}
                with self._stats_lock:
                    self.frames_detected += 1
                    self.detect_busy_s += busy
                    for k in self.yolo_speed_s:
                        if speed.get(k) is not None:
                            self.yolo_speed_s[k] += speed[k] / 1000.0
                self._publish(seq, (frame_idx, frame, result))
        except BaseException as e:
            self._fail(e)

    def _publish(self, seq: int, item: tuple) -> None:
        with self._done_cv:
            self._done[seq] = item
            self._done_cv.notify_all()

    def __iter__(self):
        seq = 0
        while True:
            t0 = time.perf_counter()
            with self._done_cv:
                while seq not in self._done and self._error is None and \
                        not (self._total is not None and seq >= self._total):
                    self._done_cv.wait(timeout=_POLL)
                if self._error is not None:
                    raise self._error
                item = self._done.pop(seq, None)
            self.wait_s += time.perf_counter() - t0
            if item is None:
                return
            self._slots.release()
            self.frames_yielded += 1
            yield item
            seq += 1

    def close(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=30)
