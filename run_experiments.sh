#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# Run GFlowNet training for all step fractions.
#
# Each step_fraction gets its own sweep across seeds and learning rates.
# Precomputed reward tables must exist at the paths below.
#
# Total runs: 3 step_fractions × 3 seeds × 3 learning_rates = 27
# Expected time per run: ~2-5 minutes (cached rewards)
# Total wall time: ~1-2 hours sequential, or ~minutes with --multirun
# ─────────────────────────────────────────────────────────────────

REWARD_DIR="precomputed"

for SF in 0.10 0.15 0.30; do
    echo "=== Running step_fraction=${SF} ==="

    REWARD_CACHE_PATH="${REWARD_DIR}/reward_table_sf${SF}.json" \
    python train.py --multirun \
        seed=0,1,2 \
        gflownet.optimizer.lr=1e-4,5e-4,1e-3 \
        logger.project_name="gfn-crop-calibration" \
        logger.tags="[sf_${SF}]" \
        2>&1 | tee "logs/sweep_sf${SF}.log"

    echo ""
done

echo "All experiments complete."
