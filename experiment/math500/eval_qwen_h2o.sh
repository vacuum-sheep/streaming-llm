#!/bin/bash

# MATH500 Evaluation Script for Qwen Models
# This script runs the MATH500 benchmark with different configurations for Qwen models

# Default settings
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATA_ROOT="data"
OUTPUT_DIR="experiment/math500/qwen_result_h2o"
NUM_SAMPLES=500

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Running MATH500 Evaluation for Qwen Models"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES"
echo ""


echo "Running streaming evaluation with start_size=4, recent_size=2048..."
CUDA_VISIBLE_DEVICES=0 nohup python experiment/math500/run_math_evaluation.py \
    --model_name_or_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --enable_streaming \
    --start_size 4 \
    --recent_size 256 \
    --hh_size 256 \
    --num_samples $NUM_SAMPLES \
    --max_gen_len 8000 > eval_streaming_h2o_qwen.log 2>&1 &

