#!/bin/bash

# MATH500 Evaluation Script
# This script runs the MATH500 benchmark with different configurations

# Default settings
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DATA_ROOT="../../data/"
OUTPUT_DIR="math500/results/"
NUM_SAMPLES=50  # Set to None to run all 500 questions

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
echo "Running dense evaluation..."
python run_math_evaluation.py \
    --model_name_or_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --num_samples $NUM_SAMPLES \
    --max_gen_len 512

echo ""

# Run streaming evaluation with different configurations
echo "Running streaming evaluation with start_size=4, recent_size=2048..."
python run_math_evaluation.py \
    --model_name_or_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --enable_streaming \
    --start_size 4 \
    --recent_size 2048 \
    --num_samples $NUM_SAMPLES \
    --max_gen_len 512

echo ""

echo "Running streaming evaluation with start_size=4, recent_size=1024..."
python run_math_evaluation.py \
    --model_name_or_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --enable_streaming \
    --start_size 4 \
    --recent_size 1024 \
    --num_samples $NUM_SAMPLES \
    --max_gen_len 512

echo ""

echo "Running streaming evaluation with start_size=4, recent_size=512..."
python run_math_evaluation.py \
    --model_name_or_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --enable_streaming \
    --start_size 4 \
    --recent_size 512 \
    --num_samples $NUM_SAMPLES \
    --max_gen_len 512

echo ""
echo "All evaluations completed!"
echo "Results saved in: $OUTPUT_DIR" 
