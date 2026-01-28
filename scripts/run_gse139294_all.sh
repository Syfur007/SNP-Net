#!/bin/bash
# Run all GSE139294 experiments sequentially
# Usage: bash scripts/run_gse139294_all.sh

set -e  # Exit on error

# Activate environment if needed
if [ -d "env/bin" ]; then
    source env/bin/activate
fi

echo "========================================"
echo "Running All GSE139294 Experiments"
echo "========================================"
echo ""

experiments=(
    "GSE139294_dpcformer"
    "GSE139294_wheatgp"
    "GSE139294_bilstm"
    "GSE139294_transformer_cnn"
    "GSE139294_vae"
)

total=${#experiments[@]}
current=1

for exp in "${experiments[@]}"; do
    echo "[$current/$total] Running experiment: $exp"
    echo "----------------------------------------"
    python src/train.py experiment=$exp
    echo ""
    echo "✓ Completed: $exp"
    echo ""
    ((current++))
done

echo "========================================"
echo "✓ All GSE139294 experiments completed!"
echo "========================================"
