# Begin from Cursor Generated 

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
    Extract the final answer from the model's response.
    This is a simple extraction - you might want to improve this based on your model's output format.
    """
    # Remove common prefixes and clean up the response
    response = response.strip()
    
    # Try to find the answer in various formats
    # Look for boxed answers: \boxed{...}
    boxed_match = re.search(r'\\boxed\{([^}]*)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Look for "The answer is" or similar patterns
    answer_patterns = [
        r'the answer is[:\s]+([^\n.]+)',
        r'answer[:\s]+([^\n.]+)',
        r'result[:\s]+([^\n.]+)',
        r'final answer[:\s]+([^\n.]+)',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no specific pattern found, return the last line or last sentence
    lines = response.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('Therefore') and not line.startswith('Thus'):
            return line
    
    return response


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
    normalized_predicted = normalize_answer(predicted_answer)
    normalized_correct = normalize_answer(correct_answer)
    
    # Check if answers match
    is_correct = normalized_predicted == normalized_correct
    
    return {
        'question': question,
        'correct_answer': correct_answer,
        'predicted_answer': predicted_answer,
        'normalized_predicted': normalized_predicted,
        'normalized_correct': normalized_correct,
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
        default="../../data/",
        help="Path to the data directory containing MATH500.jsonl"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/",
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
        default=512,
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
