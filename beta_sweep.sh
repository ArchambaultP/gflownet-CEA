#!/bin/bash

FILE=${1:-precomputed/reward_table_sf0.3.json}
BETAS=(1 5 10 20 30 50 75 95 120 150 200)

echo "Beta sweep for: $FILE"
echo "========================================"

for beta in "${BETAS[@]}"; do
    echo ""
    echo "--- Beta = $beta ---"
    python analyze_reward_table.py "$FILE" --beta "$beta"
done
