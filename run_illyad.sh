#!/bin/bash
set -e
cd /home/users/oauser/mvsa

start_cam() {
    local SESSION_NAME="$1"
    local FOLDER_ID="$2"
    local CAM_NAME="$3"
    local DEVICE="$4"
    local DB_PATH="historical-processor/data/manifest_${CAM_NAME}.db"
    local OUT_DIR="historical-processor/output/${CAM_NAME}"
    local CFG_PATH="vehicle-counting/config/vehicle_count_config_${CAM_NAME}.yaml"

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux kill-session -t "$SESSION_NAME"
    fi

    echo "Launching $CAM_NAME on GPU $DEVICE"
    tmux new-session -d -s "$SESSION_NAME" \
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID && export CUDA_VISIBLE_DEVICES=$DEVICE && env/bin/python historical-processor/orchestrator.py \
            --folder-id '$FOLDER_ID' \
            --camera-name '$CAM_NAME' \
            --db '$DB_PATH' \
            --output-dir '$OUT_DIR' \
            --config '$CFG_PATH' \
            --dl-threads 1 \
            --inf-threads 2 \
            --scan"
}

start_cam "cam_cvc3_cam_1" "1AQx8un5dlHqAuzYgFU431NrVffifrEhJ" "cvc3_cam_1" "0"
start_cam "cam_cvc4_cam_1" "1g3QIxoRLpq-OwkriQzeVSgWjR9sX51Km" "cvc4_cam_1" "0"
start_cam "cam_cvc5_cam_1__ir" "11ZlSp86bXwmKGl1MFrlh_FG37UXlEkio" "cvc5_cam_1__ir" "0"
start_cam "cam_cvc5_cam_2" "1fMNFeKSUhsHIzqBPfrLj85fnraX8aqyQ" "cvc5_cam_2" "0"
start_cam "cam_cvc6_cam_1__nb" "1I79IJc28D3d4G0k7ueCxFBf7pZI3a4Cs" "cvc6_cam_1__nb" "0"
start_cam "cam_cvc6_cam_2__sb" "15wCqQmf-MkYIA9y2D6YGJSfhHkWZUb9a" "cvc6_cam_2__sb" "0"

echo "Illyad cameras launched!"
