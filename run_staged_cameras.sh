#!/bin/bash
# Staged camera processing: runs 4 cameras at a time on the shared GPU so each
# batch gets ~1/4 of the H100 instead of 1/11th, finishing in ~1-2 days per
# batch instead of ~2 weeks if all 11 ran together.
cd /home/users/oauser/mvsa

start_cam() {
    local SESSION_NAME="$1"
    local FOLDER_ID="$2"
    local CAM_NAME="$3"
    local DB_PATH="historical-processor/data/manifest_${CAM_NAME}.db"
    local OUT_DIR="historical-processor/output/${CAM_NAME}"
    local CFG_PATH="vehicle-counting/config/vehicle_count_config_${CAM_NAME}.yaml"

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux kill-session -t "$SESSION_NAME"
    fi

    echo "[+] Launching $CAM_NAME in tmux session '$SESSION_NAME'..."
    tmux new-session -d -s "$SESSION_NAME" \
        "/home/users/oauser/mvsa/env/bin/python historical-processor/orchestrator.py \
            --folder-id '$FOLDER_ID' \
            --camera-name '$CAM_NAME' \
            --db '$DB_PATH' \
            --output-dir '$OUT_DIR' \
            --config '$CFG_PATH' \
            --dl-threads 4 \
            --inf-threads 1 \
            --scan"
}

wait_for_batch() {
    # Waits until none of the given tmux sessions exist anymore
    # (orchestrator.py exits its loop once "All files processed").
    local sessions=("$@")
    while true; do
        local any_alive=0
        for s in "${sessions[@]}"; do
            if tmux has-session -t "$s" 2>/dev/null; then
                any_alive=1
                break
            fi
        done
        if [ "$any_alive" -eq 0 ]; then
            break
        fi
        sleep 60
    done
}

echo "===== BATCH 1: tvc5_cam2_wb, tvc2_cam1, tvc3_cam1, tvc4_cam1_eb ====="
start_cam "cam_tvc5_wb"   "14INUpnOHhrVy1t3_JQkWtrIEpRr3ZgCE" "tvc5_cam2_wb"
start_cam "cam_tvc2_cam1" "1uo3kO-dOtae2kovrp8hhO-8kvObfssZy" "tvc2_cam1"
start_cam "cam_tvc3_cam1" "1DQZXCRdauo8IEcNecenXGbiKUzmgp-67" "tvc3_cam1"
start_cam "cam_tvc4_eb"   "1iOHBQWn89TXU2OrcPBEa-K9zKQ8tlsiP" "tvc4_cam1_eb"
wait_for_batch "cam_tvc5_wb" "cam_tvc2_cam1" "cam_tvc3_cam1" "cam_tvc4_eb"
echo "===== BATCH 1 COMPLETE ====="

echo "===== BATCH 2: tvc4_cam2_wb, tvc5_cam1_eb, tvc6_cam1_nb, tvc6_cam2_sb ====="
start_cam "cam_tvc4_wb" "1VcDjYGSFV_JA6V3BtjVK2pTLV_drX9m5" "tvc4_cam2_wb"
start_cam "cam_tvc5_eb" "1fHpvoRMCz0xFK-pWZm_CyW1tTGAQkWmB" "tvc5_cam1_eb"
start_cam "cam_tvc6_nb" "1UhELkjqKlG-OcsAqrnHN9HgYet3e3zFZ" "tvc6_cam1_nb"
start_cam "cam_tvc6_sb" "1_GidlTKc-Kvp-8hILi90quMEOOOaSpHF" "tvc6_cam2_sb"
wait_for_batch "cam_tvc4_wb" "cam_tvc5_eb" "cam_tvc6_nb" "cam_tvc6_sb"
echo "===== BATCH 2 COMPLETE ====="

echo "===== BATCH 3: tvc7_cam1_nb, tvc7_cam2_sb, tvc8_cam1 ====="
start_cam "cam_tvc7_nb"   "18Oi-fyV8w5MPqfO4v12WDT6stcohhDho" "tvc7_cam1_nb"
start_cam "cam_tvc7_sb"   "1Akg2gJd_bBLqfEMZ3kK-0xnf0RfFg1sU" "tvc7_cam2_sb"
start_cam "cam_tvc8_cam1" "1iT_LjMtQoyIFam_Akl2eJvs0Q0c80xav" "tvc8_cam1"
wait_for_batch "cam_tvc7_nb" "cam_tvc7_sb" "cam_tvc8_cam1"
echo "===== BATCH 3 COMPLETE ====="

echo "===== ALL 11 CAMERAS FULLY PROCESSED ====="
