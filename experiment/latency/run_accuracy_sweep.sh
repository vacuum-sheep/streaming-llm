#!/bin/bash

MODEL=huggyllama/llama-7b
TOKENS=2048
START_SIZE=4
OUTPUT_CSV=streaming_latency_llama7b.csv

echo "recent_size,latency_ms" > $OUTPUT_CSV

for RECENT_SIZE in 256 512 1024 2048 4096
do
  echo "=== Running StreamingLLM: recent_size=$RECENT_SIZE ==="
  
  LATENCY=$(python experiment/latency/run_latency_eval.py \
    --model_name_or_path $MODEL \
    --num_tokens $TOKENS \
    --enable_streaming \
    --start_size $START_SIZE \
    --recent_size $RECENT_SIZE \
    2>/dev/null | grep "Average latency" | awk '{print $(NF-1)}')

  echo "$RECENT_SIZE,$LATENCY" >> $OUTPUT_CSV
done
