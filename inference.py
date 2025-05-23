#!/usr/bin/env python
"""
Simple inference script for DeepSeek paraphrase model.
Usage: python inference.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def load_model(model_name="PeterAM4/deepseek-paraphrase", device="cuda"):
    """Load model and tokenizer"""
    print(f"Loading model {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    print(f"Model loaded on {device}")
    return model, tokenizer

def paraphrase_text(text, model, tokenizer, temperature=0.7, max_length=200):
    """Generate paraphrase for input text"""
    
    prompt = f"""<｜begin▁of▁sentence｜><｜User｜>Paraphrase the following text while\
        preserving its meaning but changing the wording and structure: {text}"""
    
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    print(inputs, type(inputs))
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    paraphrase = full_response.replace(prompt, "").strip()
    
    return paraphrase

def main():
    """Main interactive loop"""
    
    # Load model
    model, tokenizer = load_model()
    
    print("\n" + "="*60)
    print("DeepSeek Paraphrase Model - Interactive Mode")
    print("="*60)
    print("Enter text to paraphrase (or 'quit' to exit)")
    print("Press Enter twice to submit multi-line text\n")
    
    while True:
        # Get user input
        print("\nOriginal text:")
        lines = []
        while True:
            line = input()
            if line.lower() == 'quit':
                print("Exiting...")
                sys.exit(0)
            if line == "" and lines:  # Empty line after text
                break
            if line:
                lines.append(line)
        
        if not lines:
            continue
            
        text = " ".join(lines)
        
        # Generate paraphrase
        print("\nGenerating paraphrase...")
        try:
            paraphrase = paraphrase_text(text, model, tokenizer)
            
            print("\nParaphrase:")
            print("-" * 40)
            print(paraphrase)
            print("-" * 40)
            
        except Exception as e:
            print(f"Error generating paraphrase: {e}")

if __name__ == "__main__":
    main()