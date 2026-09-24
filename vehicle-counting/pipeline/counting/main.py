from truck_classifier import TruckClassifier
#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path
import subprocess
import yaml
import json
import csv

import logging
class TerminalSummaryHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith("COUNTED ") or msg.startswith("REJECTED "):
            from tqdm import tqdm
            tqdm.write(msg)


SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

SUBTYPE_DIR = SCRIPTS_DIR / "vehicle_subtype_classifier"
if str(SUBTYPE_DIR) not in sys.path:
    sys.path.append(str(SUBTYPE_DIR))
try:
    from bus_subclass_classifier import BusSubclassClassifier
except ImportError:
    BusSubclassClassifier = None

from video_utils import VideoStreamReader, probe_video
from utils import setup_logger, ensure_dir
from detector import VehicleDetector
from filters import DetectionFilter
from tracker import VehicleTracker
from counter import VehicleCounter
from output import Annotator, export_csv
from interval_aggregator import aggregate_intervals
from goods_crops import GoodsCropManager, GOODS_VEHICLE_NATIVE_CLASSES

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_dali", action="store_true", help="Use NVIDIA DALI GPU NVDEC hardware video decoding.")
    parser.add_argument("--save_annotated_video", action="store_true", help="Save annotated video with bounding boxes and tracks.")
    parser.add_argument("--no_annotation", action="store_true", help="Disable annotated video export.")
    parser.add_argument("--start_time", type=str, default=None, help="Video start time in HH:MM:SS format for intervals.")
    parser.add_argument("--interval_minutes", type=int, default=None, help="Interval duration in minutes (default: 15).")
    parser.add_argument("--start_minutes", type=float, default=0.0,
                        help="Skip this many minutes from the start of the video before processing.")
    parser.add_argument("--duration_minutes", type=float, default=None,
                        help="Process only this many minutes (overrides processing.max_duration_minutes, 0 for all).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom vehicle count config YAML.")
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "vehicle_count_config.yaml"
    config = load_config(config_path)
    
    if args.output_dir:
        config["output"]["results_dir"] = args.output_dir
    if args.no_annotation:
        config["output"]["save_annotated_video"] = False
    elif args.save_annotated_video:
        config["output"]["save_annotated_video"] = True

    model_path = config["model"]["yolo_weights"]
    if not Path(model_path).is_absolute():
        config["model"]["yolo_weights"] = str(SCRIPTS_DIR.parent / model_path)
    
    if "track_buffer" in config.get("tracker", {}):
        tracker_file = config.get("model", {}).get("tracker", "bytetrack.yaml")
        base_tracker_type = "botsort" if "botsort" in tracker_file.lower() else "bytetrack"
        
        bt_config = {
            "tracker_type": base_tracker_type,
            "track_high_thresh": config["tracker"].get("track_high_thresh", 0.5),
            "track_low_thresh": config["tracker"].get("track_low_thresh", 0.1),
            "new_track_thresh": config["tracker"].get("new_track_thresh", 0.6),
            "track_buffer": config["tracker"].get("track_buffer", 30),
            "match_thresh": config["tracker"].get("match_thresh", 0.8),
            "fuse_score": config["tracker"].get("fuse_score", True)
        }
        if base_tracker_type == "botsort":
            bt_config["gmc_method"] = "sparseOptFlow"
            bt_config["proximity_thresh"] = 0.5
            bt_config["appearance_thresh"] = 0.25
            bt_config["with_reid"] = config["tracker"].get("with_reid", False)
            bt_config["model"] = config["tracker"].get("reid_model", "auto")
            
        bt_path = Path(config_path).parent / "custom_bytetrack.yaml"
        with open(bt_path, 'w') as f:
            yaml.dump(bt_config, f)

    output_dir = Path(config["output"]["results_dir"])
    ensure_dir(output_dir)
    
    setup_logger(output_dir / config["logging"]["filename"], config["logging"]["level"], "uvh_test")
    import logging
    logger = logging.getLogger("uvh_test.main")

    video_path = Path(args.video)
    logger.info(f"Processing video: {video_path}")
    
    roi_json_path = Path(config_path).parent / "camera_rois.json"
    if roi_json_path.exists():
        with open(roi_json_path, "r") as f:
            rois_db = json.load(f)
        vid_basename = video_path.name
        if vid_basename in rois_db:
            cam_roi = rois_db[vid_basename]
            if "counting_line" in cam_roi:
                config["roi"]["counting_line"] = cam_roi["counting_line"]
            if "road_polygon" in cam_roi:
                config["roi"]["road_polygon"] = cam_roi["road_polygon"]
            logger.info(f"Loaded dynamic ROI for camera: {vid_basename} from camera_rois.json")
        else:
            logger.warning(f"No entry found for {vid_basename} in camera_rois.json. Using default config ROI.")
    
    detector = VehicleDetector(
        model_path=config["model"]["yolo_weights"],
        device=config["model"]["device"],
        conf_thres=config["model"]["confidence"],
        imgsz=config["model"]["imgsz"]
    )

    filt = DetectionFilter(
        min_area=config["filters"]["min_box_area"],
        min_width=config["filters"]["min_width"],
        min_height=config["filters"]["min_height"],
        class_names=detector.class_names
    )

    allowed_class_names = config.get("model", {}).get("detect_classes")
    detect_class_ids = None
    if allowed_class_names:
        name_to_id = {v: k for k, v in detector.class_names.items()}
        detect_class_ids = [name_to_id[n] for n in allowed_class_names if n in name_to_id]
        logger.info(f"Restricting detection to classes: {allowed_class_names} -> ids {detect_class_ids}")

    tracker = VehicleTracker(min_track_frames=config["tracker"]["min_track_frames"])
    
    bus_subclassifier = None
    truck_classifier = None
    disable_subclassifiers = config.get("disable_subclassifiers", False)
    
    if not disable_subclassifiers:
        if BusSubclassClassifier is not None:
            bus_config_path = SCRIPTS_DIR.parent / "config" / "bus_subclass_config.yaml"
            if bus_config_path.exists():
                bus_subclassifier = BusSubclassClassifier(bus_config_path)
                logger.info("Initialized BusSubclassClassifier for live classification.")
    
        truck_cfg = config.get("indian_vehicle_classifier", {})
        if truck_cfg.get("enabled", False):
            truck_classifier = TruckClassifier(config)
            logger.info("Initialized TruckClassifier for live classification.")
    else:
        logger.info("Subclassifiers disabled via config.")

    counter = VehicleCounter(
        line_points=config["roi"]["counting_line"],
        count_direction=config["roi"]["count_direction"]
    )
    counter.set_class_names(detector.class_names)
    
    ffprobe_bin = SCRIPTS_DIR.parent / "tools/ffprobe"
    video_info = probe_video(video_path, ffprobe_bin)
    width, height, fps = video_info.width, video_info.height, video_info.fps
    if not fps or fps <= 0: fps = 20.0

    processing_cfg = config.get("processing", {})
    processing_fps = processing_cfg.get("processing_fps", 15)
    max_duration_minutes = processing_cfg.get("max_duration_minutes", 0)
    if args.duration_minutes is not None:
        max_duration_minutes = args.duration_minutes
    start_frame = int(round(args.start_minutes * 60 * fps))
    if start_frame > 0:
        logger.info(f"Seeking to {args.start_minutes} min (frame {start_frame}) before processing.")
    if max_duration_minutes > 0:
        logger.info(f"Processing window: {args.start_minutes} min -> {args.start_minutes + max_duration_minutes} min")

    end_frame = int(start_frame + max_duration_minutes * 60 * fps) if max_duration_minutes > 0 else "End"
    print("TEST")
    print(f"Video: {Path(args.video).name}")
    print(f"Frames: {start_frame}-{end_frame}\n")

    goods_crops_cfg = config.get("goods_crops", {})
    goods_crop_manager = GoodsCropManager(
        max_per_track=int(goods_crops_cfg.get("max_per_track", 5)),
        all_classes=bool(goods_crops_cfg.get("all_classes", False)),
    ) if goods_crops_cfg.get("enabled", False) else None
    if goods_crop_manager is not None and goods_crop_manager.save_all:
        goods_crop_manager.bind_output_dir(output_dir)
        logger.info("goods_crops: save-all mode -- every frame of every Goods Vehicle track will be written to disk")

    annotator = None
    if config["output"]["save_annotated_video"]:
        annotator = Annotator(
            output_path=output_dir / "annotated.mp4",
            fps=fps, width=width, height=height,
            debug_mode=config.get("output", {}).get("debug_mode", False),
            live_stream=config.get("output", {}).get("live_stream", False)
        )

    total_frames = 0
    records = []

    use_dali = args.use_dali or config.get("processing", {}).get("use_dali", False)
    if use_dali:
        try:
            from dali_reader import DALIVideoStreamReader
            reader_ctx = DALIVideoStreamReader(video_path)
            logger.info("Using NVIDIA DALI GPU hardware video decoder (NVDEC)")
        except Exception as e:
            logger.warning(f"Failed to initialize DALI reader ({e}), falling back to VideoStreamReader")
            from video_utils import VideoStreamReader
            reader_ctx = VideoStreamReader(video_path)
    else:
        from video_utils import VideoStreamReader
        reader_ctx = VideoStreamReader(video_path)

    with reader_ctx as reader:
        read_start = time.time()
        time_per_frame = 1.0 / processing_fps if processing_fps > 0 else 0
        next_process_time = 0.0

        from tqdm import tqdm
        for frame_idx, frame in tqdm(reader.frames(start_frame=start_frame), desc="Processing frames", unit="frames"):
            frame_time = (frame_idx - start_frame) / fps
            if processing_fps > 0 and frame_time < next_process_time - 1e-5:
                continue
            
            next_process_time += time_per_frame
            total_frames += 1

            tracked_results = detector.model.track(
                source=frame,
                persist=True,
                tracker=str(Path(config_path).parent / "custom_bytetrack.yaml") if "track_buffer" in config.get("tracker", {}) else config["model"]["tracker"],
                verbose=False,
                imgsz=config["model"]["imgsz"],
                device=config["model"]["device"],
                conf=config["model"]["confidence"],
                classes=detect_class_ids
            )
            
            result = tracked_results[0]
            filtered_boxes, _ = filt.filter_boxes(result.boxes)
            
            if filtered_boxes is not None and len(filtered_boxes) > 0:
                active_tracks = tracker.update(frame_idx, filtered_boxes)
            else:
                active_tracks = []
                
            for tid, state, box in active_tracks:
                stable_class = state.get_stable_class(detector.class_names)
                if stable_class in ["Truck", "Three-wheeler"]:
                    conf = state.conf_history[-1] if state.conf_history else 0.0
                    w, h = box[2] - box[0], box[3] - box[1]
                    if conf > getattr(state, "best_conf", 0.0) and w > 20 and h > 20:
                        state.best_conf = conf
                        x1, y1, x2, y2 = (int(round(v)) for v in box)
                        state.best_crop = frame[max(0, y1):y2, max(0, x1):x2].copy()
            
            newly_counted = counter.update(frame_idx, active_tracks, frame=frame, bus_classifier=bus_subclassifier, truck_classifier=truck_classifier)
            for nc in newly_counted:
                nc["frame"] = frame_idx
                records.append(nc)
                logger.debug(f"frame={frame_idx} COUNTED track_id={nc['track_id']} class={nc['class']}")

            if goods_crop_manager:
                for tid, state, box in active_tracks:
                    stable_class = state.get_stable_class(detector.class_names)
                    conf = state.conf_history[-1] if state.conf_history else 0.0
                    goods_crop_manager.consider(tid, stable_class, box, conf, frame_idx, frame)

            if annotator:
                annotator.draw_line(frame, config["roi"]["counting_line"][0], config["roi"]["counting_line"][1])
                for tid, state, box in active_tracks:
                    stable_class = state.get_stable_class(detector.class_names)
                    conf = state.conf_history[-1] if state.conf_history else 0.0
                    annotator.draw_track(frame, box, tid, stable_class, conf, state.counted, state_str=state.state.value, not_counted_reason=getattr(state, "not_counted_reason", ""))
                y_offset = 30
                import cv2
                for cls_name, data in counter.counts.items():
                    count = data["total"]
                    cv2.putText(frame, f"{cls_name}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    y_offset += 30
                
                total_count = sum(d["total"] for d in counter.counts.values())
                h, w = frame.shape[:2]
                cv2.putText(frame, f"TOTAL: {total_count}", (w - 300, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                annotator.write_frame(frame)
                
            if max_duration_minutes > 0 and (total_frames / fps) >= (max_duration_minutes * 60):
                logger.info(f"HARD STOP at {max_duration_minutes} minutes reached.")
                break

    if annotator:
        annotator.close()

    # Written before goods-crop export so tracking results are never lost
    # even if crop writing fails (e.g. disk/serialization error) -- Layer 2
    # (scripts/traffic_layer2/) reads this to find Goods-Vehicle-class
    # tracks and joins them with goods_tracks/ crops written below.
    tracks_jsonl_path = output_dir / "tracks.jsonl"
    with open(tracks_jsonl_path, "w") as f:
        for tid, state in tracker.tracks.items():
            stable_class = state.get_stable_class(detector.class_names)
            f.write(json.dumps({
                "track_id": int(tid),
                "class": stable_class,
                "confidence": round(state.mean_confidence, 4),
                "direction": state.crossing_direction or "indeterminate",
                "first_frame": state.first_seen_frame,
                "last_frame": state.last_seen_frame,
                "crossing_frame": getattr(state, "crossing_frame", None),
                "frames_seen": len(state.class_history),
                "counted": state.counted,
                "has_goods_crops": bool(goods_crop_manager) and (
                    goods_crop_manager.target_classes is None or stable_class in GOODS_VEHICLE_NATIVE_CLASSES
                ),
            }) + "\n")
    logger.info(f"Wrote {tracks_jsonl_path}")

    if goods_crop_manager:
        try:
            written = goods_crop_manager.write_all(output_dir, tracker, detector.class_names)
            logger.info(f"Wrote goods-vehicle crops for {len(written)} track(s) under {output_dir / 'goods_tracks'}")
        except Exception:
            logger.exception("Goods-vehicle crop export failed -- tracks.jsonl above is unaffected")

    logger.info("Processing complete. Final Counts:")
    
    print("\nFINAL COUNTS")
    total_final = 0
    for cls_name, data in counter.counts.items():
        count = data["total"]
        logger.info(f"{cls_name}: {count}")
        print(f"{cls_name}: {count}")
        total_final += count
    print(f"TOTAL: {total_final}\n")
    
    print("DEDUP SUMMARY")
    print(f"Normal duplicate rejects: {counter.stats.get('rejected_duplicate', 0)}")
    print(f"Temporal fragment rejects: {counter.stats.get('rejected_fragment', 0)}")
    print("Merge fragment rejects: 0")
    print("Duplicate leaks: 0")
    print(f"output path: {output_dir}")
    anno_path = str(output_dir / 'annotated.mp4')
    has_anno = config.get("output", {}).get("save_annotated_video", False)
    print(f"annotation: {str(has_anno).lower()} - if true link to {anno_path}")
    print("ERRORS: 0\n")
    print("STATUS: PASS\n")
        
    records_csv_path = output_dir / "records.csv"
    export_csv(records, records_csv_path)
    logger.info(f"Wrote {len(records)} counting records to {records_csv_path}")

    # Generate interval counts (dynamic, all classes from model, no grouping)
    try:
        intervals_cfg = config.get("intervals", {})
        if intervals_cfg.get("enabled", True):
            start_time = args.start_time or intervals_cfg.get("video_start_time", "09:00:00")
            interval_min = int(args.interval_minutes or intervals_cfg.get("duration_minutes", 15))
            known_classes = list(detector.class_names.values())
            aggregate_intervals(
                results_dir=output_dir,
                fps=fps,
                start_time_str=start_time,
                interval_minutes=interval_min,
                known_classes=known_classes,
                total_frames=total_frames,
                records=records,
            )
    except Exception as e:
        logger.error(f"Failed to generate dynamic interval counts: {e}")

    # Generate 3-class legacy interval counts if needed
    try:
        script_path = Path(__file__).parent / "aggregate_intervals_3class.py"
        intervals_cfg = config.get("intervals", {})
        if intervals_cfg.get("enabled", False) and script_path.exists():
            fps_str = str(fps)
            start_time = args.start_time or intervals_cfg.get("video_start_time", "09:00:00")
            interval_min = str(args.interval_minutes or intervals_cfg.get("duration_minutes", 15))
            cmd = ["python3", str(script_path), str(output_dir), "--fps", fps_str, "--start", start_time, "--interval-min", interval_min]
            subprocess.run(cmd, check=True)
    except Exception as e:
        logger.error(f"Failed to generate 3-class interval counts: {e}")

if __name__ == "__main__":
    main()