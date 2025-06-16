#!/bin/bash

# 通用参数
MODEL=huggyllama/llama-7b
DATASET_NAME=pg19
TASK=pg19
SPLIT=test
OUTPUT_ROOT=experiment/fig3/llama7b
START_SIZE=4
RECENT_SIZE=2044

# 创建输出根目录
mkdir -p $OUTPUT_ROOT

# 不同的 num_eval_tokens 设置
for NUM_EVAL_TOKENS in 5000 10000 15000 20000
do
  echo "=== StreamingLLM (Tokens=$NUM_EVAL_TOKENS) ==="
  python examples/eval_long_ppl.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET_NAME \
    --task $TASK \
    --split $SPLIT \
    --enable_start_recent_kv_cache \
    --start_size $START_SIZE \
    --recent_size $RECENT_SIZE \
    --num_eval_tokens $NUM_EVAL_TOKENS \
    --output_dir ${OUTPUT_ROOT}/streaming_${NUM_EVAL_TOKENS}

  echo "=== Dense (Tokens=$NUM_EVAL_TOKENS) ==="
  python examples/eval_long_ppl.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET_NAME \
    --task $TASK \
    --split $SPLIT \
    --num_eval_tokens $NUM_EVAL_TOKENS \
    --output_dir ${OUTPUT_ROOT}/dense_${NUM_EVAL_TOKENS}

  echo "=== Window (Tokens=$NUM_EVAL_TOKENS) ==="
  python examples/eval_long_ppl.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET_NAME \
    --task $TASK \
    --split $SPLIT \
    --enable_start_recent_kv_cache \
    --start_size 0 \
    --recent_size 2048 \
    --num_eval_tokens $NUM_EVAL_TOKENS \
    --output_dir ${OUTPUT_ROOT}/window_${NUM_EVAL_TOKENS}
done
