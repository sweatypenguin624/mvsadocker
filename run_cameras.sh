#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

TARGET="${1:-all}"

start_cam() {
    local SESSION_NAME="$1"
    local FOLDER_ID="$2"
    local CAM_NAME="$3"
    local DB_PATH="historical-processor/data/manifest_${CAM_NAME}.db"
    local OUT_DIR="historical-processor/output/${CAM_NAME}"
    local CFG_PATH="vehicle-counting/config/vehicle_count_config_${CAM_NAME}.yaml"

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "[!] tmux session '$SESSION_NAME' already exists. Killing previous session to restart cleanly..."
        tmux kill-session -t "$SESSION_NAME"
    fi

    echo "[+] Launching $CAM_NAME in tmux session '$SESSION_NAME'..."
    echo "    Config: $CFG_PATH"
    echo "    DB:     $DB_PATH"
    echo "    Output: $OUT_DIR"

    tmux new-session -d -s "$SESSION_NAME" \
        "/home/users/oauser/mvsa/env/bin/python historical-processor/orchestrator.py \
            --folder-id '$FOLDER_ID' \
            --camera-name '$CAM_NAME' \
            --db '$DB_PATH' \
            --output-dir '$OUT_DIR' \
            --config '$CFG_PATH' \
            --dl-threads 4 \
            --inf-threads 2"
    echo "[✓] $CAM_NAME started in background (tmux session: $SESSION_NAME)"
}

case "$TARGET" in
    wb|tvc5_wb)
        start_cam "camWB" "14INUpnOHhrVy1t3_JQkWtrIEpRr3ZgCE" "tvc5_cam2_wb"
        ;;
    nb|tvc1_nb)
        start_cam "camAgriNB" "1uM9Pad4IvLipkwyTF41i1iGDor2S7wjN" "tvc1_cam1_nb"
        ;;
    sb|tvc1_sb)
        start_cam "camAgriSB" "1iHJvxpv-9MzutTtVgE1-j_dG3fcHQ3xP" "tvc1_cam2_sb"
        ;;
    agri|agriculture)
        start_cam "camAgriNB" "1uM9Pad4IvLipkwyTF41i1iGDor2S7wjN" "tvc1_cam1_nb"
        start_cam "camAgriSB" "1iHJvxpv-9MzutTtVgE1-j_dG3fcHQ3xP" "tvc1_cam2_sb"
        ;;
    all)
        start_cam "camWB" "14INUpnOHhrVy1t3_JQkWtrIEpRr3ZgCE" "tvc5_cam2_wb"
        start_cam "camAgriNB" "1uM9Pad4IvLipkwyTF41i1iGDor2S7wjN" "tvc1_cam1_nb"
        start_cam "camAgriSB" "1iHJvxpv-9MzutTtVgE1-j_dG3fcHQ3xP" "tvc1_cam2_sb"
        ;;
    *)
        echo "Usage: ./run_cameras.sh [all | wb | nb | sb | agri]"
        exit 1
        ;;
esac

echo ""
echo "Active tmux sessions:"
tmux ls
