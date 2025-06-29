#!/bin/bash

# MATH500 Evaluation Script
# This script runs the MATH500 benchmark with different configurations

# Default settings
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DATA_ROOT="data"
OUTPUT_DIR="experiment/math500/results/"
NUM_SAMPLES=500

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Running MATH500 Evaluation"
echo "=========================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES"
echo ""

# Run dense evaluation (no streaming)
# echo "Running dense evaluation..."
# CUDA_VISIBLE_DEVICES=0 nohup python experiment/math500/run_math_evaluation.py \
#     --model_name_or_path $MODEL_PATH \
#     --data_root $DATA_ROOT \
#     --output_dir $OUTPUT_DIR \
#     --num_samples $NUM_SAMPLES \
#     --max_gen_len 8000  > eval_dense.log 2>&1 &

# echo ""

# # Run streaming evaluation with different configurations
# echo "Running streaming evaluation with start_size=4, recent_size=2048..."
# CUDA_VISIBLE_DEVICES=1 nohup python experiment/math500/run_math_evaluation.py \
#     --model_name_or_path $MODEL_PATH \
#     --data_root $DATA_ROOT \
#     --output_dir $OUTPUT_DIR \
#     --enable_streaming \
#     --start_size 4 \
#     --recent_size 2048 \
#     --num_samples $NUM_SAMPLES \
#     --max_gen_len 8000 > eval_streaming_2048.log 2>&1 &

# echo ""

# echo "Running streaming evaluation with start_size=4, recent_size=1024..."
# CUDA_VISIBLE_DEVICES=0 nohup python run_m   ath_evaluation.py \
#     --model_name_or_path $MODEL_PATH \
#     --data_root $DATA_ROOT \
#     --output_dir $OUTPUT_DIR \
#     --enable_streaming \
#     --start_size 4 \
#     --recent_size 1024 \
#     --num_samples $NUM_SAMPLES \
#     --max_gen_len 8000 > eval_streaming_1024.log 2>&1 &

# echo ""

# echo "Running streaming evaluation with start_size=4, recent_size=512..."
CUDA_VISIBLE_DEVICES=1 nohup python experiment/math500/run_math_evaluation.py \
    --model_name_or_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --enable_streaming \
    --start_size 4 \
    --recent_size 512 \
    --num_samples $NUM_SAMPLES \
    --max_gen_len 8000 > eval_streaming_512.log 2>&1 &

# echo ""
# echo "All evaluations completed!"
# echo "Results saved in: $OUTPUT_DIR" 
