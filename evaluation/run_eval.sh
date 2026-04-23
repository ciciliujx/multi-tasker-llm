#!/bin/bash
# Evaluate a trained checkpoint on all 3 tasks and save submission.json.
#
# Usage:
#   bash evaluation/run_eval.sh <checkpoint_path> [limit] [temperature] [top_p]
#
# Examples:
#   bash evaluation/run_eval.sh "tinker://3eaa8c2d-...:train:0/sampler_weights/demo"
#   bash evaluation/run_eval.sh "tinker://3eaa8c2d-...:train:0/sampler_weights/demo" 20
#   bash evaluation/run_eval.sh "tinker://3eaa8c2d-...:train:0/sampler_weights/demo" 0 0.3 0.9

set -e

CHECKPOINT_PATH="${1:?Usage: bash evaluation/run_eval.sh <checkpoint_path> [limit] [temperature] [top_p]}"
LIMIT="${2:-5}"
TEMPERATURE="${3:-0.0}"
TOP_P="${4:-1.0}"

echo "========================================"
echo "Checkpoint:  $CHECKPOINT_PATH"
echo "Samples:     $LIMIT (0 = full eval)"
echo "Temperature: $TEMPERATURE"
echo "Top-p:       $TOP_P"
echo "========================================"

python evaluation/eval_all.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --limit "$LIMIT" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P"

echo ""
echo "Done. Submission saved to evaluation/submission.json"
