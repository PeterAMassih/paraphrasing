#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune DeepSeek-R1-Distill-Qwen-7B for paraphrasing with LoRA + TRL SFTTrainer.
"""
import os
import random
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig  # Import SFTConfig


def set_random_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def train_paraphrase_model(resume_from_checkpoint=None):
    # Set seed for reproducibility
    set_random_seed(42)
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,                      # Rank dimension for LoRA
        lora_alpha=32,             # Alpha parameter (2x rank)
        lora_dropout=0.05,         # Dropout for regularization
        task_type="CAUSAL_LM",     # Task type for causal language models
        target_modules=[
            # Attention modules
            "q_proj", "k_proj", "v_proj", "o_proj",
            # MLP modules
            "gate_proj", "down_proj",
        ],
        bias="none",               # Freeze all original biases
        modules_to_save=["lm_head"],  # Keep lm_head trainable for vocabulary diversity
        use_rslora=True,           # Enable rank-stabilized LoRA
    )
    
    print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_ds = load_dataset("json", data_files="data/deepseek/deepseek_train.json")["train"]
    eval_ds = load_dataset("json", data_files="data/deepseek/deepseek_val.json")["train"]
    print(f"Training dataset size: {len(train_ds):,}")
    print(f"Evaluation dataset size: {len(eval_ds):,}")

    output_dir = "deepseek-paraphrase-lora"
    os.makedirs(output_dir, exist_ok=True)

    per_device_bs = 4              
    grad_acc_steps = 4             # Gradient accumulation steps
    effective_batch_size = per_device_bs * grad_acc_steps
    
    epochs = 3                     # Number of training epochs
    total_steps = (len(train_ds) * epochs) // effective_batch_size
    save_steps = max(total_steps // 10, 1)    # Save approximately 10 checkpoints
    eval_steps = max(total_steps // 20, 1)    # Evaluate approximately 20 times

    # Create SFTConfig with all the SFT-specific parameters
    sft_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_acc_steps,
        num_train_epochs=epochs,
        learning_rate=2e-4,             
        lr_scheduler_type="cosine",     
        warmup_ratio=0.05,              
        weight_decay=0.01,              
        max_grad_norm=0.3,              
        bf16=True,                      
        logging_steps=50,               
        save_strategy="steps",          
        save_steps=save_steps,
        save_total_limit=3,             
        eval_strategy="steps",     # Use eval_strategy instead of evaluation_strategy
        eval_steps=eval_steps,
        load_best_model_at_end=True,    
        metric_for_best_model="eval_loss",  
        report_to="tensorboard",        
        remove_unused_columns=False,    
        gradient_checkpointing=False,   
        max_seq_length=1000,            # Max sequence length
        packing=False,                  # Don't pack sequences
    )


    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_args,              # Use SFTConfig instance
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
    )


    print(f"Starting fine-tuning{' from checkpoint' if resume_from_checkpoint else ''}...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = os.path.join(output_dir, "final")
    print(f"Saving adapter to {final_dir}")
    trainer.save_model(final_dir)
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a paraphrasing model with LoRA")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="Path to checkpoint directory or 'latest' to use the latest checkpoint")
    args = parser.parse_args()
    
    train_paraphrase_model(args.resume_from_checkpoint)