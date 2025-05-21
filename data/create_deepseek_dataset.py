import json
import random
import os
import torch
from datasets import load_dataset

# Set a fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def create_sft_datasets():
    """Create training and validation datasets in the correct format for TRL SFTTrainer"""
    
    # Make sure output directory exists
    os.makedirs("deepseek", exist_ok=True)
    
    # 1. Load your paraphrase_train.json
    with open("evaluation/splits/paraphrase_train.json", "r") as f:
        paraphrase_train_data = json.load(f)
    
    # 2. Load Quora dataset
    print("Loading Quora dataset...")
    quora_dataset = load_dataset("quora")
    
    # 3. Filter for positive pairs (paraphrases) in Quora
    positive_pairs = [
        item for item in quora_dataset["train"] 
        if item["is_duplicate"] == 1
    ]
    
    # 4. Sample 2000 pairs from Quora
    sampled_quora = random.sample(positive_pairs, 2000)
    print(f"Sampled {len(sampled_quora)} pairs from Quora")
    
    # 5. Create an empty list for the SFT training data
    sft_training_data = []
    
    # 6. Process paraphrase_train.json (all 3 paraphrases per sample)
    paraphrase_count = 0
    print("Processing your paraphrase dataset...")
    for category in paraphrase_train_data:
        for item in paraphrase_train_data[category]:
            original = item["text"]
            for paraphrase in item["paraphrases"]:
                # Format with proper thinking pattern
                sft_training_data.append({
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Paraphrase the following text while preserving its meaning but changing the wording and structure: {original}"
                        },
                        {
                            "role": "assistant",
                            "content": f"<think>\nLet me analyze this text and find ways to rephrase it while keeping the same meaning.\nI need to use different vocabulary and structure.\n</think>\n\n{paraphrase}"
                        }
                    ]
                })
                paraphrase_count += 1
    
    print(f"Added {paraphrase_count} samples from your dataset")
    
    # 7. Process Quora samples
    print("Processing Quora samples...")
    for item in sampled_quora:
        # Get the question pair
        original = item["questions"]["text"][0]
        paraphrase = item["questions"]["text"][1]
        
        # Add to training data with same thinking pattern
        sft_training_data.append({
            "messages": [
                {
                    "role": "user",
                    "content": f"Paraphrase the following text while preserving its meaning but changing the wording and structure: {original}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nLet me analyze this text and find ways to rephrase it while keeping the same meaning.\nI need to use different vocabulary and structure.\n</think>\n\n{paraphrase}"
                }
            ]
        })
    
    print(f"Total samples in SFT training dataset: {len(sft_training_data)}")
    
    # 8. Save the training dataset
    with open("deepseek/deepseek_train.json", "w") as f:
        json.dump(sft_training_data, f, indent=2, ensure_ascii=False)
    
    print("Training dataset saved to deepseek/deepseek_train.json")
    
    # 9. Create a validation set from paraphrase_test.json
    print("Processing test set for validation data...")
    with open("evaluation/splits/paraphrase_test.json", "r") as f:
        test_data = json.load(f)
    
    validation_data = []
    val_count = 0
    
    for category in test_data:
        for item in test_data[category]:
            original = item["text"]
            # Just use the first paraphrase for validation
            if item["paraphrases"]:
                paraphrase = item["paraphrases"][0]
                validation_data.append({
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Paraphrase the following text while preserving its meaning but changing the wording and structure: {original}"
                        },
                        {
                            "role": "assistant",
                            "content": f"<think>\nLet me analyze this text and find ways to rephrase it while keeping the same meaning.\nI need to use different vocabulary and structure.\n</think>\n\n{paraphrase}"
                        }
                    ]
                })
                val_count += 1
    
    # 10. Add 100 unseen Quora samples to validation
    unseen_quora = [
        item for item in quora_dataset["train"] 
        if item["is_duplicate"] == 1 and item not in sampled_quora
    ]
    
    validation_quora = random.sample(unseen_quora, min(100, len(unseen_quora)))
    
    for item in validation_quora:
        original = item["questions"]["text"][0]
        paraphrase = item["questions"]["text"][1]
        
        validation_data.append({
            "messages": [
                {
                    "role": "user",
                    "content": f"Paraphrase the following text while preserving its meaning but changing the wording and structure: {original}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nLet me analyze this text and find ways to rephrase it while keeping the same meaning.\nI need to use different vocabulary and structure.\n</think>\n\n{paraphrase}"
                }
            ]
        })
    
    print(f"Created validation set with {len(validation_data)} samples")
    
    # 11. Save validation data
    with open("deepseek/deepseek_val.json", "w") as f:
        json.dump(validation_data, f, indent=2, ensure_ascii=False)
    
    print("Validation dataset saved to deepseek/deepseek_val.json")
    print("Both datasets are ready for TRL SFTTrainer!")

if __name__ == "__main__":
    create_sft_datasets()