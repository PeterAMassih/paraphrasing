"""
Script to merge the LoRA adapter with the base model, rename, and push to Hugging Face Hub.
"""

import os
import argparse
import torch
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo

def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model and push to Hugging Face Hub")
    parser.add_argument("--lora_path", type=str, default="deepseek-paraphrase-lora/final", 
                        help="Path to the LoRA adapter")
    parser.add_argument("--base_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
                        help="Base model name or path")
    parser.add_argument("--output_name", type=str, default="deepseek-paraphrase", 
                        help="Output directory name for merged model")
    parser.add_argument("--hf_repo_name", type=str, default="PeterAM4/deepseek-paraphrase", 
                        help="Hugging Face Hub repository name (username/repo_name)")
    parser.add_argument("--hf_token", type=str, required=True, 
                        help="Hugging Face auth token with write permissions")
    parser.add_argument("--private", action="store_true", 
                        help="Make the repository private")
    
    return parser.parse_args()

def create_model_card(model_name, base_model, repo_name):
    """Create a basic model card for the merged model"""
    return f"""---
language:
- en
tags:
- deepseek
- paraphrase
- lora
- text-generation
license: mit
datasets:
- quora
model-index:
- name: {model_name}
  results: []
---

# {model_name}

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) that has been specialized for high-quality paraphrase generation. It was trained using LoRA (Low-Rank Adaptation) and then merged back into the base model for efficient inference.

## Model Details

- **Base Model**: {base_model}
- **Task**: Paraphrase Generation
- **Training Method**: LoRA fine-tuning with r=16, alpha=32
- **Training Data**: Multi-domain text from literary works, technical documentation, academic papers, and articles, plus the Quora paraphrase dataset

## Performance

This model outperforms standard paraphrasing models like BART and T5 on key metrics:
- **Semantic Preservation** (BERTScore): 0.952 - Excellent
- **Lexical Diversity** (BLEU Diversity): 0.513 - Acceptable
- **Character-level Changes** (Edit Distance): 0.344 - Acceptable
- **Structural Variation** (Syntactic Diversity): 0.147 - Moderate
- **Overall Balance** (Harmonic Score): 0.468 - Acceptable

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "{repo_name}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

text = "Learn Once, Write Anywhere: We don't make assumptions about the rest of your technology stack, so you can develop new features in React without rewriting existing code."

prompt = f"<｜begin▁of▁sentence｜><｜User｜>Paraphrase the following text while preserving its meaning but changing the wording and structure: {{text}}<｜Assistant｜><think>\\nLet me analyze this text and find ways to rephrase it while keeping the same meaning.\\nI need to use different vocabulary and structure.\\n</think>\\n\\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.95,
    do_sample=True
)

paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "")
print(paraphrase)
```

## Limitations

- Very technical or domain-specific terminology may not be paraphrased optimally
- Always review paraphrases for factual accuracy and meaning preservation

## Citation

If you use this model in your research or applications, please cite:

```
@misc{{deepseek-paraphrase,
  author = {{PeterAM4}},
  title = {{DeepSeek Paraphrase: Fine-tuned DeepSeek model for high-quality paraphrasing}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{repo_name}}}}}
}}
```
"""

def main():
    args = parse_args()
    
    print(f"Starting LoRA merging process...")
    print(f"LoRA adapter path: {args.lora_path}")
    print(f"Base model: {args.base_model}")
    
    os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token
    
    os.makedirs(args.output_name, exist_ok=True)
    
    # Method 1: Use AutoPeftModelForCausalLM to load and merge in one step
    try:
        print("Loading and merging model (Method 1)...")
        
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.lora_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        print("Merging adapter weights with base model...")
        merged_model = model.merge_and_unload()
        
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        
    except Exception as e:
        print(f"Method 1 failed with error: {e}")
        print("Trying Method 2...")
        
        # Method 2: Load base model and adapter separately, then merge
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
            
            print("Loading adapter...")
            model = PeftModel.from_pretrained(base_model, args.lora_path)
            
            print("Merging adapter weights with base model...")
            merged_model = model.merge_and_unload()
            
        except Exception as e2:
            print(f"Method 2 also failed with error: {e2}")
            raise RuntimeError("Could not merge LoRA weights. Please check your model paths and dependencies.")
    
    print(f"Successfully merged model. Saving to {args.output_name}...")
    
    merged_model.save_pretrained(args.output_name)
    tokenizer.save_pretrained(args.output_name)
    
    print("Model and tokenizer saved locally.")
    
    model_card = create_model_card(
        model_name=args.output_name.replace('-', ' ').title(),
        base_model=args.base_model,
        repo_name=args.hf_repo_name
    )
    
    with open(os.path.join(args.output_name, "README.md"), "w") as f:
        f.write(model_card)
    
    print("Model card created.")
    
    print(f"Creating or updating repository {args.hf_repo_name}...")
    api = HfApi(token=args.hf_token)
    
    try:
        repo_url = create_repo(
            repo_id=args.hf_repo_name,
            token=args.hf_token,
            private=args.private,
            exist_ok=True
        )
        print(f"Repository URL: {repo_url}")
    except Exception as e:
        print(f"Note: {e}")
        print("Continuing with upload...")
    
    print("Uploading model to Hugging Face Hub. This may take a while...")
    api.upload_folder(
        folder_path=args.output_name,
        repo_id=args.hf_repo_name,
        token=args.hf_token
    )
    
    print(f"Successfully uploaded model to {args.hf_repo_name}!")
    print(f"View your model at: https://huggingface.co/{args.hf_repo_name}")
    
if __name__ == "__main__":
    main()