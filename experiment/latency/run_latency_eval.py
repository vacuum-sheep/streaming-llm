import time
import torch
import argparse
from streaming_llm.utils import load
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from datasets import load_dataset

@torch.no_grad()
def measure_latency(model, tokenizer, args):
    ds = load_dataset("pg19", split="test", streaming=True)
    first_book = next(iter(ds))["text"]
    prompt = first_book[:20000]  # 取前两万字符（约6000–8000 tokens）
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Prepare initial KV
    past_key_values = None
    if args.enable_streaming:
        kv_cache = enable_streaming_llm( # kv_cache: the kvcache strategy oracle
            model,
            start_size=args.start_size,
            recent_size=args.recent_size
        )
    else:
        kv_cache = None

    # First pass (prefill)
    outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    if kv_cache:
        past_key_values = kv_cache(past_key_values)

    # Greedy decoding with timing
    latencies = []
    next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    for _ in range(args.num_tokens):
        start = time.time()
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        torch.cuda.synchronize()
        end = time.time()
        latencies.append(end - start)

        past_key_values = outputs.past_key_values
        if kv_cache:
            past_key_values = kv_cache(past_key_values)

        next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    avg_latency = sum(latencies) / len(latencies) * 1000
    print(f"Average latency over {args.num_tokens} tokens: {avg_latency:.2f} ms")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--num_tokens", type=int, default=2048)
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2048)
    args = parser.parse_args()

    model, tokenizer = load(args.model_name_or_path)
    measure_latency(model, tokenizer, args)

if __name__ == "__main__":
    main()
