#!/usr/bin/env python3

import torch
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(__file__))

from streaming_llm.utils import load

def test_simple_streaming():
    """Test simple streaming generation without custom patches"""
    print("Testing simple streaming generation...")
    
    # Load model
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model, tokenizer = load(model_name)
    
    # Simple prompt
    prompt = "What is 2 + 2? Answer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    print(f"Input prompt: {prompt}")
    print(f"Input tokens: {input_ids[0].tolist()}")
    
    # Generate with simple streaming
    with torch.no_grad():
        # Initial forward pass
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        
        # Get the first token
        next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids = [next_token.item()]
        
        print(f"First token: {next_token.item()} -> '{tokenizer.decode([next_token.item()], skip_special_tokens=True)}'")
        
        # Generate remaining tokens
        for step in range(20):
            # Forward pass with the next token
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            
            # Get the next token
            next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids.append(next_token.item())
            
            token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            print(f"Step {step+1}: token {next_token.item()} -> '{token_text}'")
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Full response: {response}")
    print("="*50)

def test_math_prompt_simple():
    """Test with math prompt using simple streaming"""
    print("Testing with math prompt using simple streaming...")
    
    # Load model
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model, tokenizer = load(model_name)
    
    # Math prompt
    question = "What is 2 + 2?"
    prompt = f"Problem: {question}\nSolve this problem. Put your final answer within \\boxed{{}}\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    print(f"Input prompt: {prompt}")
    print(f"Input tokens: {input_ids[0].tolist()}")
    
    # Generate with simple streaming
    with torch.no_grad():
        # Initial forward pass
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        
        # Get the first token
        next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids = [next_token.item()]
        
        print(f"First token: {next_token.item()} -> '{tokenizer.decode([next_token.item()], skip_special_tokens=True)}'")
        
        # Generate remaining tokens
        for step in range(50):
            # Forward pass with the next token
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            
            # Get the next token
            next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids.append(next_token.item())
            
            if step < 10:  # Only print first 10 steps
                token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                print(f"Step {step+1}: token {next_token.item()} -> '{token_text}'")
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Check for repetitive tokens
            if len(generated_ids) > 10:
                recent_tokens = generated_ids[-10:]
                if len(set(recent_tokens)) == 1:
                    print(f"Warning: Detected repetitive token generation at step {step+1}")
                    break
    
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Full response: {response}")
    print("="*50)

if __name__ == "__main__":
    print("Starting simple streaming tests...")
    print("="*50)
    
    try:
        test_simple_streaming()
    except Exception as e:
        print(f"Error in simple streaming: {e}")
    
    try:
        test_math_prompt_simple()
    except Exception as e:
        print(f"Error in math prompt simple: {e}")
    
    print("Simple streaming tests completed.") 
