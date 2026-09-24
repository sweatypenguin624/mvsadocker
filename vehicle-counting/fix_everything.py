import re
import os

MAIN_PATH = "/home/users/oauser/mvsa/vehicle-counting/pipeline/counting/main.py"
with open(MAIN_PATH, "r") as f:
    main_code = f.read()

# Fix VehicleCounter initialization
main_code = re.sub(
    r'counter = VehicleCounter\(\s*line_pt1=config\["roi"\]\["counting_line"\]\[0\],\s*line_pt2=config\["roi"\]\["counting_line"\]\[1\],\s*direction=config\["roi"\]\["count_direction"\]\s*\)',
    'counter = VehicleCounter(\n        line_points=config["roi"]["counting_line"],\n        count_direction=config["roi"]["count_direction"]\n    )\n    counter.set_class_names(detector.class_names)',
    main_code
)

# Fix counter.process_tracks -> counter.update
main_code = main_code.replace(
    'newly_counted = counter.process_tracks(active_tracks, detector.class_names)',
    'newly_counted = counter.update(frame_idx, active_tracks)'
)

# Fix counter.counts_by_class -> counter.counts
main_code = main_code.replace(
    'for cls_name, count in counter.counts_by_class.items():',
    'for cls_name, data in counter.counts.items():\n                    count = data["total"]'
)
main_code = main_code.replace(
    'total_count = sum(counter.counts_by_class.values())',
    'total_count = sum(d["total"] for d in counter.counts.values())'
)

with open(MAIN_PATH, "w") as f:
    f.write(main_code)
print("main.py fixed")


COUNTER_PATH = "/home/users/oauser/mvsa/vehicle-counting/pipeline/counting/counter.py"
with open(COUNTER_PATH, "r") as f:
    counter_code = f.read()

# To handle ByteTrack ID fragmentation conservatively, we check if a newly crossed track
# is too close (spatially and temporally) to a recently counted track of the same class & direction.
# We add a `recent_crossings` list to VehicleCounter.

if "recent_crossings" not in counter_code:
    # Add recent_crossings to __init__
    counter_code = counter_code.replace(
        'self.counted_ids = set()',
        'self.counted_ids = set()\n        self.recent_crossings = []  # stores (frame_idx, direction, class, point)'
    )
    
    # Add to reset
    counter_code = counter_code.replace(
        'self.counted_ids.clear()',
        'self.counted_ids.clear()\n        self.recent_crossings.clear()'
    )
    
    # Add anti-fragmentation logic in update() before counting it valid
    fragment_check = """
            # --- Anti-Fragmentation (Conservative) ---
            # If a track of the same class crossed recently in the same direction, and is very close,
            # this might be the same physical vehicle with a broken Track ID.
            is_fragment = False
            for rc_frame, rc_dir, rc_class, rc_point in self.recent_crossings:
                if rc_class == stable_class and rc_dir == direction:
                    if frame_idx - rc_frame < 60: # within 2 seconds (assuming 30fps)
                        dist = ((point[0] - rc_point[0])**2 + (point[1] - rc_point[1])**2)**0.5
                        if dist < 400.0: # spatial threshold
                            is_fragment = True
                            self.stats["rejected_duplicate"] += 1
                            logger.info(f"REJECTED: frame={frame_idx} tid={tid} as fragment of recent crossing.")
                            break
            
            if is_fragment:
                # Add to counted_ids anyway so we don't keep checking it
                self.counted_ids.add(tid)
                continue
            
            # Record this valid crossing
            self.recent_crossings.append((frame_idx, direction, stable_class, point))
            # Keep only recent crossings to avoid memory bloat
            self.recent_crossings = [rc for rc in self.recent_crossings if frame_idx - rc[0] < 300]
            
            self.counted_ids.add(tid)"""
            
    counter_code = counter_code.replace(
        'self.counted_ids.add(tid)',
        fragment_check,
        1 # Only replace the first occurrence in the valid count section
    )
    
    with open(COUNTER_PATH, "w") as f:
        f.write(counter_code)
    print("counter.py fixed for ID fragmentation")

