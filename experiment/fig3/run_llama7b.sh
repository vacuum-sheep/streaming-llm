#!/bin/bash

# 设置通用参数
MODEL=huggyllama/llama-7b
DATASET_NAME=pg19
TASK=pg19
SPLIT=test
NUM_EVAL_TOKENS=5000
OUTPUT_ROOT=experiment/llama7b
mkdir -p $OUTPUT_ROOT



#  echo "=== Dense Attention ==="
# python examples/eval_long_ppl.py \
#   --model_name_or_path $MODEL \
#   --dataset_name $DATASET_NAME \
#   --task $TASK \
#   --split $SPLIT \
#   --num_eval_tokens $NUM_EVAL_TOKENS \
#   --output_dir ${OUTPUT_ROOT}/dense



# echo "=== Window Attention ==="
# python examples/eval_long_ppl.py \
#   --model_name_or_path $MODEL \
#   --dataset_name $DATASET_NAME \
#   --task $TASK \
#   --split $SPLIT \
#   --enable_start_recent_kv_cache \
#   --start_size 0 \
#   --recent_size 2048 \
#   --num_eval_tokens $NUM_EVAL_TOKENS \
#   --output_dir ${OUTPUT_ROOT}/window



echo "=== StreamingLLM (Start+Recent) ==="
python examples/eval_long_ppl.py \
  --model_name_or_path $MODEL \
  --dataset_name $DATASET_NAME \
  --task $TASK \
  --split $SPLIT \
  --enable_start_recent_kv_cache \
  --start_size 4 \
  --recent_size 2044 \
  --num_eval_tokens $NUM_EVAL_TOKENS \
  --output_dir ${OUTPUT_ROOT}/streaming
