#!/bin/bash
set -e
export WANDB_MODE=online
REWARD_DIR="precomputed"
for SF in 0.15 0.3 0.4; do
    for SEED in 0 1 2; do
        for LR in 1e-4 5e-4 1e-3; do
            echo "=== SF=${SF} SEED=${SEED} LR=${LR} ==="
            LOGFILE="logs/sweep_sf${SF}_s${SEED}_lr${LR}.log"
            REWARD_CACHE_PATH="${REWARD_DIR}/reward_table_sf${SF}.json" \
            python train.py \
                seed=${SEED} \
                gflownet.optimizer.lr=${LR} \
                proxy.reward_cache_path="${REWARD_DIR}/reward_table_sf${SF}.json" \
                logger.project_name="gfn-crop-calibration" \
                >> "$LOGFILE" 2>&1
            echo "Done. Log: $LOGFILE"
        done
    done
done
echo "All experiments complete."
