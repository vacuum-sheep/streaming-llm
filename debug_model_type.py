#!/usr/bin/env python3

import torch
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(__file__))

from streaming_llm.utils import load

def check_model_type():
    """Check the model type and attention classes"""
    print("Checking model type and attention classes...")
    
    # Load model
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model, tokenizer = load(model_name)
    
    print(f"Model type: {model.config.model_type}")
    print(f"Model class: {type(model).__name__}")
    
    # Find attention modules more specifically
    attention_modules = []
    for name, module in model.named_modules():
        if 'self_attn' in name.lower() and hasattr(module, 'forward'):
            attention_modules.append((name, type(module).__name__))
    
    print(f"\nFound {len(attention_modules)} attention modules:")
    for name, module_type in attention_modules[:10]:  # Show first 10
        print(f"  {name}: {module_type}")
    
    # Check if it's a Qwen2 model
    if hasattr(model.config, 'model_type'):
        print(f"\nModel config type: {model.config.model_type}")
        if 'qwen' in model.config.model_type.lower():
            print("This appears to be a Qwen model")
        else:
            print("This is NOT a Qwen model")
    
    # Check for specific attention classes
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
        qwen2_attentions = []
        
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                qwen2_attentions.append(name)
        
        print(f"\nQwen2Attention modules: {len(qwen2_attentions)}")
        if qwen2_attentions:
            print("Found Qwen2Attention modules - using Qwen2 position shift")
            print(f"First few: {qwen2_attentions[:3]}")
        else:
            print("No Qwen2Attention modules found")
    except ImportError as e:
        print(f"Could not import Qwen2Attention: {e}")
    
    # Let's also check what the actual attention class is
    if attention_modules:
        first_attn_name, first_attn_class = attention_modules[0]
        print(f"\nFirst attention module: {first_attn_name} -> {first_attn_class}")
        
        # Get the actual module
        attn_module = None
        for name, module in model.named_modules():
            if name == first_attn_name:
                attn_module = module
                break
        
        if attn_module:
            print(f"Module attributes: {dir(attn_module)[:20]}")  # First 20 attributes
            print(f"Module base classes: {type(attn_module).__mro__}")

if __name__ == "__main__":
    check_model_type() 
