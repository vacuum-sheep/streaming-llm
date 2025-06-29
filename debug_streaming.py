#!/usr/bin/env python3

import torch
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(__file__))

from streaming_llm.utils import load
from streaming_llm.enable_streaming_llm import enable_streaming_llm

def test_simple_generation():
    """Test simple generation without streaming first"""
    print("Testing simple generation without streaming...")
    
    # Load model
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model, tokenizer = load(model_name)
    
    # Simple prompt
    prompt = "Hello, how are you?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    print(f"Input prompt: {prompt}")
    print(f"Input tokens: {input_ids[0].tolist()}")
    
    # Generate without streaming
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    print("="*50)

def test_streaming_generation():
    """Test streaming generation with correct position_ids"""
    print("Testing streaming generation...")
    
    # Load model
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model, tokenizer = load(model_name)
    
    # Enable streaming
    kv_cache = enable_streaming_llm(model, start_size=4, recent_size=2048)
    
    # Simple prompt
    prompt = "Hello, how are you?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    print(f"Input prompt: {prompt}")
    print(f"Input tokens: {input_ids[0].tolist()}")
    
    # Generate with streaming
    with torch.no_grad():
        # Initial forward pass with the prompt
        outputs = model(
            input_ids=input_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        
        # Get the first token to generate
        next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids = [next_token.item()]
        
        # Track the current position
        current_position = input_ids.shape[1]  # Start after the prompt
        
        print(f"First token: {next_token.item()} -> '{tokenizer.decode([next_token.item()], skip_special_tokens=True)}'")
        
        # Generate remaining tokens
        for step in range(20):
            # Apply KV cache eviction if enabled
            past_key_values = kv_cache.evict_for_space(past_key_values, 1)
            
            # Create position_ids for the current token
            position_ids = torch.tensor([[current_position]], device=model.device)
            
            # Forward pass with the next token and correct position_ids
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                position_ids=position_ids,
            )
            past_key_values = outputs.past_key_values
            
            # Get the next token
            next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids.append(next_token.item())
            
            # Increment position for next iteration
            current_position += 1
            
            token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            print(f"Step {step+1}: token {next_token.item()} -> '{token_text}' (position: {current_position-1})")
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Full response: {response}")
    print("="*50)

def test_math_prompt():
    """Test with math prompt"""
    print("Testing with math prompt...")
    
    # Load model
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model, tokenizer = load(model_name)
    
    # Enable streaming
    kv_cache = enable_streaming_llm(model, start_size=4, recent_size=2048)
    
    # Math prompt
    question = "What is 2 + 2?"
    prompt = f"Problem: {question}\nSolve this problem. Put your final answer within \\boxed{{}}\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    print(f"Input prompt: {prompt}")
    print(f"Input tokens: {input_ids[0].tolist()}")
    
    # Generate with streaming
    with torch.no_grad():
        # Initial forward pass with the prompt
        outputs = model(
            input_ids=input_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        
        # Get the first token to generate
        next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids = [next_token.item()]
        
        # Track the current position
        current_position = input_ids.shape[1]  # Start after the prompt
        
        print(f"First token: {next_token.item()} -> '{tokenizer.decode([next_token.item()], skip_special_tokens=True)}'")
        
        # Generate remaining tokens
        for step in range(50):
            # Apply KV cache eviction if enabled
            past_key_values = kv_cache.evict_for_space(past_key_values, 1)
            
            # Create position_ids for the current token
            position_ids = torch.tensor([[current_position]], device=model.device)
            
            # Forward pass with the next token and correct position_ids
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                position_ids=position_ids,
            )
            past_key_values = outputs.past_key_values
            
            # Get the next token
            next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids.append(next_token.item())
            
            # Increment position for next iteration
            current_position += 1
            
            if step < 10:  # Only print first 10 steps
                token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                print(f"Step {step+1}: token {next_token.item()} -> '{token_text}' (position: {current_position-1})")
            
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
    print("Starting debug tests...")
    print("="*50)
    
    try:
        test_simple_generation()
    except Exception as e:
        print(f"Error in simple generation: {e}")
    
    try:
        test_streaming_generation()
    except Exception as e:
        print(f"Error in streaming generation: {e}")
    
    try:
        test_math_prompt()
    except Exception as e:
        print(f"Error in math prompt: {e}")
    
    print("Debug tests completed.") 
