import torch
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.utils import parse_args, load

device = "cuda"

args = parse_args()

print("Arguments:")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")
    
# 动态生成文件名前缀
setting_tag = ""
if args.enable_start_recent_kv_cache:
    setting_tag += f"_start{args.start_size}_recent{args.recent_size}"
if args.enable_pos_shift:
    setting_tag += "_posshift"
if setting_tag == "":
    setting_tag = "_dense"

log_file_path = os.path.join(args.output_dir, f"log{setting_tag}.txt")
ppl_file_path = os.path.join(args.output_dir, f"ppl{setting_tag}.txt")


# data = load_dataset(args.dataset_name, args.task, split=args.split, trust_remote_code=True)

# Streaming 模式：不会下载整个数据集
stream_data = load_dataset("pg19", split="test", streaming=True, trust_remote_code=True)

# 只取第一条样本（第一本书）
data = next(iter(stream_data))

model, tokenizer = load(args.model_name_or_path)

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None

if args.enable_start_recent_kv_cache:
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "pythia" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
    else:
        raise ValueError(f"got {model.config.model_type}")
    kv_cache = StartRecentKVCache(
        start_size=args.start_size,
        recent_size=args.recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
else:
    kv_cache = None

if args.enable_pos_shift:
    if "llama" in model.config.model_type:
        from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention

        enable_llama_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    elif "gpt_neox" in model.config.model_type:
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        pass
    else:
        raise ValueError(f"got {model.config.model_type}")


os.makedirs(args.output_dir, exist_ok=True)
f = open(log_file_path, "w")

num_eval_tokens = 0
text = data["text"]
encodings = tokenizer(text, return_tensors="pt")

print(encodings.input_ids[:, :10])

seq_len = encodings.input_ids.size(1)
print(f"seq_len: {seq_len}")
pbar = tqdm(range(0, seq_len - 1))

for idx in pbar:
    input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
    with torch.no_grad():
        outputs = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits.view(-1, model.config.vocab_size)
        past_key_values = outputs.past_key_values
        label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
        neg_log_likelihood = loss_fn(logits, label)
        if kv_cache is not None:
            past_key_values = kv_cache(past_key_values)
    nlls.append(neg_log_likelihood)
    pbar.set_description(
        f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
    )
    print(neg_log_likelihood.item(), file=f, flush=True)
    num_eval_tokens += 1
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

f.close()

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(ppl_file_path, "w") as f:
    f.write(f"{ppl.item()}\n")
