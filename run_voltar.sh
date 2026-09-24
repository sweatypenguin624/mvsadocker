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

start_cam "cam_cvc7_cam_1__eb" "1hcrD2xLTRXBq80j4UrddO225W0_-Tt8q" "cvc7_cam_1__eb" "0"
start_cam "cam_cvc7_cam_2__wb" "1SgBDiTgTFTd_YltueplYmOVfXlNEC8Rq" "cvc7_cam_2__wb" "0"
start_cam "cam_cvc8_cam_1" "1Sk5D7iG8C7p_YmACEb1-O52PH8IuGkiY" "cvc8_cam_1" "0"
start_cam "cam_cvc9_cam_1" "1UIggCAMBN6S6Shw_5p3A1R1H9a5D7IlP" "cvc9_cam_1" "0"
start_cam "cam_cvc10_cam_1" "1c3YuEtGx6B8Dvz9QJfn0YxT9UwzsG5dV" "cvc10_cam_1" "0"

echo "Voltar cameras launched!"
