#!/bin/bash
# Set paths
TRANS_CHECKPOINT="checkpoints/t2m/GPT/checkpoint.pth"
OUTPUT_DIR="./generated_motions"

# Text prompts
TEXT_PROMPTS=(
    "a person puts one hand on their hip and the other in the air, then raises and lowers both arms together"
    "a man jumps twice in place"
    "a man is doing jumping jacks"
    "A man in spinning in circles and then stops"
)

# Generate motions
python generate.py \
    --trans_checkpoint "$TRANS_CHECKPOINT" \
    --text "${TEXT_PROMPTS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 10 \
    --cond_scale 2.0 \
    --min_motion_len 5 \
    --seed 42 \
    --device cuda

echo "Generation complete! Results saved to $OUTPUT_DIR"
