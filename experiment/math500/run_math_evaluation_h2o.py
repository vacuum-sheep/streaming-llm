import argparse
import json
import os.path

import tqdm
import torch
import copy
from copy import deepcopy
import dataclasses
from xopen import xopen
import math

import logging
import numpy as np

from xopen import xopen
from datetime import datetime
from typing import Union

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from utils_real_drop.modify_llama import H2OLlamaForCausalLM, H2OLlamaAttention
import re
import regex
import multiprocessing
from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

MAX_LENGTH = int(16384)  # Hardcoded max length to avoid infinite loop


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False
    
def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    # print("Judge:", prediction, reference)
    if prediction is None or reference is None:
        return False
    if str(str(prediction).strip().lower()) == str(str(reference).strip().lower()):
        return True
    # if str(prediction.strip().lower()) == str(reference.strip().lower()):
    #     return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    if "pmatrix" in prediction and not "pmatrix" in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True
    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close
        ):
            return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": None,
    "llama_h2o": H2OLlamaForCausalLM
}

TAGET_MODULE = {
    "llama": None,
    "llama_h2o": H2OLlamaAttention
}

# Placeholder for math evaluation function
def evaluate_math_answer(generated_answer, ground_truth_answer):
    pred_str = generated_answer.replace("\u043a\u0438", "")
    use_last_number = True

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        print("====================final answer is $:============================")
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        print("====================boxed:============================")
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
        print("boxed", pred)
        print("ground_truth_answer", ground_truth_answer)
        print("math_equal", math_equal(pred, ground_truth_answer))
    else:  # use the last number
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]


    return math_equal(pred, ground_truth_answer)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument("--input_path", type=str, default="/research/d1/gds/ytyang/kwchen/H2O/h2o_hf/data/MATH500.jsonl", help="Path to the dataset (jsonl format)")
    parser.add_argument("--input_path", type=str, default="data/MATH500.jsonl", help="Path to the dataset (jsonl format)")

    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    # parser.add_argument("--model_name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--hh_size", type=int, default=10)
    parser.add_argument("--recent_size", type=int, default=10)

    parser.add_argument('--enable_h2o_cache', default=False, action='store_true')

    parser.add_argument("--sample_num", type=int, default=500, help="Number of samples to evaluate from the dataset")
    parser.add_argument("--k", type=int, default=0, help="Top-k sampling parameter")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--batch_size", type=int, default=1) # Math evaluation is typically done with batch_size 1
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
        default=True
    )

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    set_seed(args)

    model_name = args.model_name
    input_path = args.input_path
    # output_path = "data/MATH500_eval_"+str(datetime.now().strftime("%Y%m%d_%H%M"))+".jsonl"
    output_path = "data/MATH500_eval_"+str(datetime.now().strftime("%Y%m%d_%H%M"))+".jsonl"
    # current_result_path = "/research/d1/gds/ytyang/kwchen/H2O/h2o_hf/data/MATH500_eval_current.jsonl"
    current_result_path = "data/MATH500_eval_current.jsonl"
    print(f"Loading model from {model_name} ...")
    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is set for batching, though bs=1 is typical

    if args.batch_size>1:
        tokenizer.pad_token = tokenizer.eos_token

    if args.enable_h2o_cache:
        print('Enabling H2O KV cache')
        config.hh_size = args.hh_size
        config.recent_size = args.recent_size
        # Ensure the H2O Llama model is compatible or adjust as needed
        model = ENABLE_Heavy_Hitter_FUNCTIONS['llama_h2o'].from_pretrained(model_name, config=config,
                                                                            cache_dir=args.cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)

    if args.fp16:
        model.half()
    model.eval().to(args.device)


    requests = []

    with xopen(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                try:
                    requests.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")


    print(f"Loaded {len(requests)} problems from {input_path}")
    if args.sample_num < len(requests) and args.sample_num > 0 :
        print(f'Sampling {args.sample_num} problems from {len(requests)} total problems')
        # Potentially use a more robust sampling method if needed, e.g., random sampling
        requests = requests[:args.sample_num]

    results = []
    correct_predictions = 0
    total_predictions = 0


    # MATH_PROMPT_TEMPLATE = "<｜begin▁of▁sentence｜><｜User｜>{question}. Please reason step by step, and put your final answer within \\boxed{{}}\n<｜end▁of▁sentence｜>\n<｜begin▁of▁sentence｜><｜Assistant｜>\n"""
    MATH_PROMPT_TEMPLATE = "Problem: {question}\nSolve this problem. Put your final answer within \\boxed{{}}\n" 

    with torch.no_grad():
        for i, request_data in enumerate(tqdm.tqdm(requests, desc="Evaluating")):

            if 'Question' not in request_data or 'Answer' not in request_data: # Ensure 'solution' or similar key exists for ground truth
                print(f"Skipping sample {i} due to missing 'Question' or 'Answer' field: {request_data}")
                continue

            problem_text = request_data['Question']

            ground_truth_answer = str(request_data['Answer']) # Ensure it's a string


            prompt = MATH_PROMPT_TEMPLATE.format(question=problem_text)

            temperature = request_data.get('temperature', 0.7) # Default to 0 for math
            top_p = request_data.get('top_p', 1.0)
            max_tokens = request_data.get('max_tokens', 8000) # Max tokens for the generated solution


            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

            # Generate output
            # For math, do_sample is often False or temperature is 0 for more deterministic output
            # You might need to adjust `max_new_tokens` instead of `max_length`
            output_sequences = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens, # More intuitive for generation length
                temperature=temperature if temperature > 0 else None, # if temp is 0, HF defaults to greedy
                top_k=args.k if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                do_sample=True if temperature > 0 else False, # Sample only if temperature > 0
                num_return_sequences=1, # Typically 1 for math evaluation
                return_dict_in_generate=True,
                output_scores=True, # If you need logprobs
                pad_token_id=tokenizer.eos_token_id, # Important for generation
                eos_token_id=tokenizer.eos_token_id # Can also use custom stop tokens here via stopping_criteria
            )

            if args.enable_h2o_cache:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE['llama_h2o']):
                        m._clean_cache()

            generated_sequence = output_sequences['sequences'].squeeze(0)
            # Decode only the generated part
            generated_text_full = tokenizer.decode(generated_sequence)
            # Remove the prompt part from the generated text
            generated_text_answer_part = tokenizer.decode(generated_sequence[len(input_ids[0]):])


            final_generated_answer = generated_text_answer_part.strip()


            is_correct = evaluate_math_answer(final_generated_answer, ground_truth_answer)
            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            # Store results
            current_result = {
                'id': request_data.get('id', i), # Assuming an 'id' field in your data, or use index
                'problem': problem_text,
                'prompt_used': prompt,
                'ground_truth_answer': ground_truth_answer,
                'generated_full_text': generated_text_full,
                'generated_answer_part': generated_text_answer_part,
                'extracted_answer': final_generated_answer, # The answer used for evaluation
                'is_correct': is_correct,
            }
            results.append(current_result)
            with xopen(current_result_path, 'a') as f:
                f.write(json.dumps(current_result) + "\n")

            # Log progress
            if (i + 1) % 10 == 0 or (i + 1) == len(requests): # Log every 10 samples or at the end
                accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
                print(f"Processed {i+1}/{len(requests)} samples. Current Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")


    # Final accuracy
    final_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"Final Evaluation Accuracy: {final_accuracy:.2f}% ({correct_predictions}/{total_predictions})")

    # Save results to output file
    args_dict = vars(args)
    if 'device' in args_dict and hasattr(args_dict['device'], '__str__'):
        args_dict['device'] = str(args_dict['device'])
    avg_num_tokens = sum([len(res['generated_full_text'].split()) for res in results]) / len(results)
    output_data = {
        "args": args_dict,
        "num_total_problems": len(requests),
        "num_evaluated_problems": total_predictions,
        "num_correct_predictions": correct_predictions,
        "accuracy": final_accuracy,
        "avg_num_tokens": avg_num_tokens,
        "results": results
    }
    with xopen(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Evaluation results saved to {output_path}") 
