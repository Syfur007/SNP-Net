#!/bin/bash
# Run all experiments in one or more subdirectories sequentially
# Usage: bash scripts/run_all.sh <subdirectory[,subdirectory,...]> [debug]

set -e  # Exit on error

# Activate environment if needed
if [ -d "env/bin" ]; then
    source env/bin/activate
fi

if [ -z "$1" ]; then
    echo "Usage: bash scripts/run_all.sh <subdirectory[,subdirectory,...]> [debug]"
    echo "Example: bash scripts/run_all.sh GSE139294"
    echo "Example: bash scripts/run_all.sh GSE139294,GSE33355"
    echo "Example: bash scripts/run_all.sh GSE139294 debug"
    exit 1
fi

IFS=',' read -r -a subdirs <<< "$1"

debug_flag=""
if [ "$2" = "debug" ]; then
    debug_flag="debug=default"
fi

for subdir in "${subdirs[@]}"; do
    subdir=$(echo "$subdir" | xargs)
    if [ -z "$subdir" ]; then
        continue
    fi

    config_dir="configs/experiment/$subdir"

    if [ ! -d "$config_dir" ]; then
        echo "Error: subdirectory not found: $config_dir"
        exit 1
    fi

    mapfile -t experiments < <(ls "$config_dir"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/\.yaml$//')

    if [ ${#experiments[@]} -eq 0 ]; then
        echo "Error: no experiment configs found in $config_dir"
        exit 1
    fi

    echo "========================================"
    echo "Running All Experiments in $subdir"
    echo "========================================"
    echo ""

    total=${#experiments[@]}
    current=1

    for exp in "${experiments[@]}"; do
        echo "[$current/$total] Running experiment: $exp"
        echo "----------------------------------------"
        python src/train.py experiment="$subdir/$exp" $debug_flag
        echo ""
        echo "✓ Completed: $exp"
        echo ""
        ((current++))
    done

    echo "========================================"
    echo "✓ All experiments completed for $subdir!"
    echo "========================================"
    echo ""
done
