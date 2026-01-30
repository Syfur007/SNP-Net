#!/bin/bash
# Run all autism experiments sequentially
# Usage: bash scripts/run_autism_all.sh

set -e  # Exit on error

# Activate environment if needed
if [ -d "env/bin" ]; then
    source env/bin/activate
fi

echo "========================================"
echo "Running All autism Experiments"
echo "========================================"
echo ""

experiments=(
    "autism_autoencoder"
    "autism_vae"
    "autism_lstm"
    "autism_bilstm"
    "autism_dpcformer"
    "autism_wheatgp"
    "autism_transformer_cnn"
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
echo "✓ All autism experiments completed!"
echo "========================================"
