#!/bin/bash
cd /home/users/oauser/mvsa
SCRATCH="/tmp/claude-15975/-storage-users-oauser-mvsa/b9d742f7-e82f-4036-a6ed-6b28d57d6483/scratchpad/verify_downloads"
mkdir -p "$SCRATCH"
REVIEW_DIR="vehicle-count-output/annotated_review"
mkdir -p "$REVIEW_DIR"

while IFS='|' read -r name fid path; do
  echo "################ $name ################"
  dav="$SCRATCH/${name}.dav"
  mp4="$SCRATCH/${name}.mp4"
  out_dir="$REVIEW_DIR/$name"
  mkdir -p "$out_dir"

  echo "[$name] downloading $path ..."
  timeout 300 ./rclone --config config/rclone.conf copyto "Gdrive-yogesh,root_folder_id=$fid:$path" "$dav" >/dev/null 2>&1 </dev/null
  if [ ! -s "$dav" ]; then
    echo "[$name] DOWNLOAD FAILED, skipping"
    continue
  fi

  echo "[$name] remuxing ..."
  tools/ffmpeg_bin/ffmpeg -y -i "$dav" -c copy "$mp4" >/dev/null 2>&1 </dev/null

  start_time=$(basename "$path" .dav | sed 's/\[R\]\[0@0\]\[0\]//' | cut -d'-' -f1 | tr '.' ':')
  echo "[$name] running 10-min annotated inference (start_time=$start_time) ..."
  env/bin/python vehicle-counting/pipeline/counting/main.py \
    --video "$mp4" \
    --output_dir "$out_dir" \
    --config "vehicle-counting/config/vehicle_count_config_${name}.yaml" \
    --save_annotated_video \
    --duration_minutes 10 \
    --start_time "$start_time" \
    > "$out_dir/run.log" 2>&1 </dev/null

  if [ -f "$out_dir/annotated.mp4" ]; then
    echo "[$name] DONE -> $out_dir/annotated.mp4"
  else
    echo "[$name] WARNING: annotated.mp4 not found, check $out_dir/run.log"
  fi

  rm -f "$dav" "$mp4"
done < "$SCRATCH/../verify_cams_remaining.txt"

echo ""
echo "===== Verification clip batch complete ====="
ls -la "$REVIEW_DIR"/*/annotated.mp4 2>&1
