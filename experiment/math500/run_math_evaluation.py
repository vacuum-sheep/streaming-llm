import warnings
warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys
from tqdm import tqdm
from typing import List, Dict, Any

# Add the parent directory to the path to import streaming_llm modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from streaming_llm.utils import load, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm


def extract_answer_from_response(response: str) -> str:
    """
    Extract the final answer from the model's response using robust logic from run_math_evaluation_h2o.py.
    """
    response = response.replace("\u043a\u0438", "")
    use_last_number = True
    import re
    # 1. If response contains "final answer is $...$. I hope"
    if "final answer is $" in response and "$. I hope" in response:
        tmp = response.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
        return pred
    # 2. If response contains "boxed"
    elif "boxed" in response:
        ans = response.split("boxed")[-1]
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
        return pred
    # 3. Otherwise, use the last number
    else:
        if use_last_number:
            pattern = r"-?\d*\.?\d+"
            pred = re.findall(pattern, response.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""
        return pred


def normalize_answer(answer: str) -> str:
    """
    Normalize the answer for comparison.
    """
    # Remove extra whitespace and convert to lowercase
    answer = re.sub(r'\s+', ' ', answer.strip()).lower()
    
    # Remove common LaTeX commands
    answer = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', answer)
    
    # Remove dollar signs
    answer = answer.replace('$', '')
    
    # Normalize fractions
    answer = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', answer)
    
    # Remove other LaTeX symbols
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)
    
    return answer


def math_equal(
    prediction,
    reference,
    include_percentage=True,
    is_close=True,
    timeout=False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    import regex
    from sympy import simplify, N
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.parsing.latex import parse_latex
    from latex2sympy2 import latex2sympy
    def numeric_equal(a, b, tol=1e-4):
        try:
            return abs(float(a) - float(b)) < tol
        except:
            return False
    def is_digit(s):
        try:
            float(s)
            return True
        except:
            return False
    def parse_digits(s):
        try:
            return float(s)
        except:
            return s
    def choice_answer_clean(s):
        s = str(s).strip().upper()
        if len(s) == 1 and s in ["A", "B", "C", "D", "E"]:
            return s
        return s
    def str_to_pmatrix(s):
        return s  # Placeholder, as matrix handling is rare
    def symbolic_equal(a, b):
        def _parse(s):
            for f in [parse_latex, parse_expr, latex2sympy]:
                try:
                    return f(s.replace("\\", "\\"))
                except:
                    try:
                        return f(s)
                    except:
                        pass
            return s
        a = _parse(a)
        b = _parse(b)
        try:
            if str(a) == str(b) or a == b:
                return True
        except:
            pass
        try:
            if hasattr(a, 'equals') and a.equals(b):
                return True
            if simplify(a - b) == 0:
                return True
        except:
            pass
        try:
            if hasattr(a, 'lhs') and hasattr(b, 'lhs'):
                if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
                    return True
        except:
            pass
        try:
            if numeric_equal(float(N(a)), float(N(b))):
                return True
        except:
            pass
        try:
            if hasattr(a, 'shape') and hasattr(b, 'shape') and a.shape == b.shape:
                _a = a.applyfunc(lambda x: round(x, 3))
                _b = b.applyfunc(lambda x: round(x, 3))
                if _a.equals(_b):
                    return True
        except:
            pass
        return False
    if prediction is None or reference is None:
        return False
    if str(str(prediction).strip().lower()) == str(str(reference).strip().lower()):
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True
    try:
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
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
    reference = str(reference).strip()
    prediction = str(prediction).strip()
    if "pmatrix" in prediction and not "pmatrix" in reference:
        reference = str_to_pmatrix(reference)
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
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all([
                math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                for i in range(len(pred_parts))
            ]):
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
        return prediction == reference
    return symbolic_equal(prediction, reference)


def evaluate_math_question(model, tokenizer, question: str, correct_answer: str, 
                          kv_cache=None, max_gen_len=512) -> Dict[str, Any]:
    """
    Evaluate a single math question.
    """
    # Create the prompt
    prompt = f"<｜begin▁of▁sentence｜><｜User｜>{question}. Please reason step by step, and put your final answer within \\boxed{{}}\n<｜end▁of▁sentence｜>\n<｜begin▁of▁sentence｜><｜Assistant｜>\n"""
    
    # Tokenize the input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=None,  # Start fresh for each question
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        
        # Generate tokens
        generated_ids = []
        for _ in range(max_gen_len):
            outputs = model(
                input_ids=torch.tensor([[generated_ids[-1] if generated_ids else outputs.logits[:, -1, :].argmax(dim=-1).item()]], 
                                     device=model.device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            
            # Apply KV cache if enabled
            if kv_cache is not None:
                past_key_values = kv_cache.evict_for_space(past_key_values, 1)
            
            next_token = outputs.logits[:, -1, :].argmax(dim=-1).item()
            generated_ids.append(next_token)
            
            if next_token == tokenizer.eos_token_id:
                break
    
    # Decode the response
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Extract and normalize answers
    predicted_answer = extract_answer_from_response(response)
    # normalized_predicted = normalize_answer(predicted_answer)
    # normalized_correct = normalize_answer(correct_answer)
    
    # Check if answers match using math_equal
    is_correct = math_equal(predicted_answer, correct_answer)
    
    return {
        'question': question,
        'correct_answer': correct_answer,
        'predicted_answer': predicted_answer,
        # 'normalized_predicted': normalized_predicted,
        # 'normalized_correct': normalized_correct,
        'is_correct': is_correct,
        'response': response
    }


def run_math_evaluation(model, tokenizer, data: List[Dict], kv_cache=None, 
                       max_gen_len=512, num_samples=None) -> Dict[str, Any]:
    """
    Run the complete MATH500 evaluation.
    """
    results = []
    correct_count = 0
    
    # Limit the number of samples if specified
    if num_samples is not None:
        data = data[:num_samples]
    
    print(f"Evaluating {len(data)} math questions...")
    
    for i, sample in enumerate(tqdm(data, desc="Evaluating")):
        question = sample['Question']
        correct_answer = sample['Answer']
        
        result = evaluate_math_question(
            model, tokenizer, question, correct_answer, 
            kv_cache, max_gen_len
        )
        
        results.append(result)
        if result['is_correct']:
            correct_count += 1
        
        # Print progress every 10 questions
        if (i + 1) % 10 == 0:
            accuracy = correct_count / (i + 1) * 100
            print(f"Progress: {i + 1}/{len(data)}, Accuracy: {accuracy:.2f}%")
    
    # Calculate final statistics
    total_questions = len(results)
    accuracy = correct_count / total_questions * 100
    
    # Group by subject and level
    subject_stats = {}
    level_stats = {}
    
    for result in results:
        # Extract subject and level from the sample
        sample_idx = results.index(result)
        subject = data[sample_idx].get('subject', 'Unknown')
        level = data[sample_idx].get('level', 'Unknown')
        
        if subject not in subject_stats:
            subject_stats[subject] = {'correct': 0, 'total': 0}
        if level not in level_stats:
            level_stats[level] = {'correct': 0, 'total': 0}
        
        subject_stats[subject]['total'] += 1
        level_stats[level]['total'] += 1
        
        if result['is_correct']:
            subject_stats[subject]['correct'] += 1
            level_stats[level]['correct'] += 1
    
    # Calculate subject and level accuracies
    subject_accuracies = {
        subject: (stats['correct'] / stats['total'] * 100) 
        for subject, stats in subject_stats.items()
    }
    level_accuracies = {
        level: (stats['correct'] / stats['total'] * 100) 
        for level, stats in level_stats.items()
    }
    
    return {
        'total_questions': total_questions,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'subject_accuracies': subject_accuracies,
        'level_accuracies': level_accuracies,
        'results': results
    }


def main(args):
    # Load model and tokenizer
    print(f"Loading model from {args.model_name_or_path} ...")
    model, tokenizer = load(args.model_name_or_path)
    
    # Load MATH500 data
    data_path = os.path.join(args.data_root, "MATH500.jsonl")
    print(f"Loading data from {data_path} ...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"MATH500 data not found at {data_path}. Please ensure the data file exists.")
    
    data = load_jsonl(data_path)
    print(f"Loaded {len(data)} questions from MATH500 dataset")
    
    # Enable streaming if requested
    kv_cache = None
    if args.enable_streaming:
        print(f"Enabling streaming with start_size={args.start_size}, recent_size={args.recent_size}")
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    
    # Run evaluation
    results = run_math_evaluation(
        model, tokenizer, data, kv_cache, 
        max_gen_len=args.max_gen_len, 
        num_samples=args.num_samples
    )
    
    # Print results
    print("\n" + "="*50)
    print("MATH500 EVALUATION RESULTS")
    print("="*50)
    print(f"Total Questions: {results['total_questions']}")
    print(f"Correct Answers: {results['correct_count']}")
    print(f"Overall Accuracy: {results['accuracy']:.2f}%")
    
    print("\nAccuracy by Subject:")
    for subject, accuracy in results['subject_accuracies'].items():
        print(f"  {subject}: {accuracy:.2f}%")
    
    print("\nAccuracy by Level:")
    for level, accuracy in results['level_accuracies'].items():
        print(f"  {level}: {accuracy:.2f}%")
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create filename based on settings
        setting_tag = ""
        if args.enable_streaming:
            setting_tag += f"_streaming_start{args.start_size}_recent{args.recent_size}"
        else:
            setting_tag += "_dense"
        
        if args.num_samples:
            setting_tag += f"_samples{args.num_samples}"
        
        # Save detailed results
        results_file = os.path.join(args.output_dir, f"math500_results{setting_tag}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
        
        # Save summary
        summary_file = os.path.join(args.output_dir, f"math500_summary{setting_tag}.txt")
        with open(summary_file, 'w') as f:
            f.write("MATH500 EVALUATION RESULTS\n")
            f.write("="*50 + "\n")
            f.write(f"Model: {args.model_name_or_path}\n")
            f.write(f"Streaming: {args.enable_streaming}\n")
            if args.enable_streaming:
                f.write(f"Start Size: {args.start_size}\n")
                f.write(f"Recent Size: {args.recent_size}\n")
            f.write(f"Total Questions: {results['total_questions']}\n")
            f.write(f"Correct Answers: {results['correct_count']}\n")
            f.write(f"Overall Accuracy: {results['accuracy']:.2f}%\n")
            
            f.write("\nAccuracy by Subject:\n")
            for subject, accuracy in results['subject_accuracies'].items():
                f.write(f"  {subject}: {accuracy:.2f}%\n")
            
            f.write("\nAccuracy by Level:\n")
            for level, accuracy in results['level_accuracies'].items():
                f.write(f"  {level}: {accuracy:.2f}%\n")
        
        print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MATH500 benchmark evaluation with streaming KV cache")
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Path to the model or model name from HuggingFace"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="data",
        help="Path to the data directory containing MATH500.jsonl"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="experiment/math500/results/",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--enable_streaming", 
        action="store_true",
        help="Enable streaming KV cache"
    )
    parser.add_argument(
        "--start_size", 
        type=int, 
        default=4,
        help="Number of tokens to keep at the start of the sequence"
    )
    parser.add_argument(
        "--recent_size", 
        type=int, 
        default=2048,
        help="Number of recent tokens to keep"
    )
    parser.add_argument(
        "--max_gen_len", 
        type=int, 
        default=8000,
        help="Maximum generation length for each question"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=None,
        help="Number of samples to evaluate (None for all)"
    )
    
    args = parser.parse_args()
    main(args) 
