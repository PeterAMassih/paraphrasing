import json
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import spacy
import pandas as pd
from tqdm import tqdm
import nltk
import os
import argparse
from peft import PeftModel
import torch

nltk.download('punkt', quiet=True)

class ParaphraseEvaluator:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Define metrics to track
        self.metrics = [
            "bertscore", 
            "bleu_diversity", 
            "edit_distance", 
            "syntactic_diversity", 
            "harmonic_score"
        ]
        
        # + 1 smoothing
        self.smoothing = SmoothingFunction().method1
        
        # N-gram weights - prioritizing lower n-grams
        self.bleu_weights = (0.4, 0.3, 0.2, 0.1)
    
    def calculate_bleu_diversity(self, original, paraphrase):
        """Calculate BLEU-based diversity with detailed n-gram breakdown"""
        original_tokens = word_tokenize(original.lower())
        paraphrase_tokens = word_tokenize(paraphrase.lower())
        
        if not original_tokens or not paraphrase_tokens:
            return {
                "bleu_score": 0.0,
                "bleu_diversity": 1.0,  # Maximum diversity if one is empty
                "bleu_1_diversity": 1.0,
                "bleu_2_diversity": 1.0,
                "bleu_3_diversity": 1.0,
                "bleu_4_diversity": 1.0
            }
        
        bleu_score = sentence_bleu(
            [original_tokens], 
            paraphrase_tokens,
            weights=self.bleu_weights,
            smoothing_function=self.smoothing
        )
        
        bleu_1 = sentence_bleu([original_tokens], paraphrase_tokens, 
                             weights=(1, 0, 0, 0), 
                             smoothing_function=self.smoothing)
        
        bleu_2 = sentence_bleu([original_tokens], paraphrase_tokens, 
                             weights=(0, 1, 0, 0), 
                             smoothing_function=self.smoothing)
        
        bleu_3 = sentence_bleu([original_tokens], paraphrase_tokens, 
                             weights=(0, 0, 1, 0), 
                             smoothing_function=self.smoothing)
        
        bleu_4 = sentence_bleu([original_tokens], paraphrase_tokens, 
                             weights=(0, 0, 0, 1), 
                             smoothing_function=self.smoothing)
        # invert
        return {
            "bleu_score": bleu_score,
            "bleu_diversity": 1.0 - bleu_score,
            "bleu_1_diversity": 1.0 - bleu_1,
            "bleu_2_diversity": 1.0 - bleu_2,
            "bleu_3_diversity": 1.0 - bleu_3,
            "bleu_4_diversity": 1.0 - bleu_4
        }
    
    def calculate_edit_distance(self, original, paraphrase):
        """Calculate normalized Levenshtein edit distance"""
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        max_len = max(len(original), len(paraphrase))
        if max_len == 0:
            return 0.0
            
        return levenshtein(original.lower(), paraphrase.lower()) / max_len
    
    def calculate_syntactic_diversity(self, original, paraphrase):
        """Calculate syntactic diversity using dependency parsing"""
        doc1 = self.nlp(original)
        doc2 = self.nlp(paraphrase)
        
        deps1 = [token.dep_ for token in doc1]
        deps2 = [token.dep_ for token in doc2]
        
        if not deps1 or not deps2:
            return 0.0
            
        overlap = len(set(deps1) & set(deps2)) / max(len(set(deps1)), len(set(deps2)))
        return 1.0 - overlap
    
    def evaluate_single(self, original, paraphrase):
        """Evaluate a single paraphrase against its original"""
        if not paraphrase or not original:
            return {metric: 0.0 for metric in self.metrics}
        
        # BERTScore for semantic similarity
        _, _, F1 = bert_score([paraphrase], [original], lang="en")
        bertscore = F1.item()
        
        # BLEU-based diversity
        bleu_results = self.calculate_bleu_diversity(original, paraphrase)
        bleu_diversity = bleu_results["bleu_diversity"]
        
        # Edit distance
        edit_distance = self.calculate_edit_distance(original, paraphrase)
        
        # Syntactic diversity
        syntactic_diversity = self.calculate_syntactic_diversity(original, paraphrase)
        
        # harm mean
        diversity_avg = (bleu_diversity + syntactic_diversity) / 2
        if bertscore > 0 and diversity_avg > 0:
            harmonic_score = 2 * (bertscore * diversity_avg) / (bertscore + diversity_avg)
        else:
            harmonic_score = 0.0
        
        return {
            "bertscore": bertscore,
            "bleu_diversity": bleu_diversity,
            "bleu_1_diversity": bleu_results["bleu_1_diversity"],
            "bleu_2_diversity": bleu_results["bleu_2_diversity"],
            "bleu_3_diversity": bleu_results["bleu_3_diversity"],
            "bleu_4_diversity": bleu_results["bleu_4_diversity"],
            "edit_distance": edit_distance,
            "syntactic_diversity": syntactic_diversity,
            "harmonic_score": harmonic_score
        }
    
    def evaluate_model(self, model_name, dataset_path, output_path=None, device="cuda"):
        """Evaluate a model on the entire dataset"""
        try:
            with open(dataset_path, 'r') as f:
                nested_dataset = json.load(f)
            
            flattened_dataset = []
            for category, items in nested_dataset.items():
                for item in items:
                    item["category"] = category
                    flattened_dataset.append(item)
            
            print(f"Successfully loaded dataset with {len(flattened_dataset)} items from {len(nested_dataset)} categories")
            
        except FileNotFoundError:
            print(f"Error: Dataset file not found: {dataset_path}")
            return None, None
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON in {dataset_path}")
            return None, None
        
        # Load model based on type
        is_lora_adapter = os.path.exists(os.path.join(model_name, "adapter_model.bin"))
        try:
            print(f"Loading model {model_name}...")
            if is_lora_adapter:
                # lora for deepsekk
                print("Detected LoRA adapter path, loading with PEFT...")
                base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
                tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    trust_remote_code=True
                )
                
                model = PeftModel.from_pretrained(base_model, model_name)
                is_seq2seq = False
                print(f"Loaded DeepSeek model with LoRA adapter")
                
            elif "bart" in model_name.lower() or "t5" in model_name.lower():
                # Seq2Seq models
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                ).to(device)
                is_seq2seq = True
                print(f"Loaded as a Seq2Seq model")
            else:
                # Causal LM models (like DeepSeek)
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                ).to(device)
                is_seq2seq = False
                print(f"Loaded as a Causal LM model")
            
            print("Model loaded successfully")
        
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
        
        results = []
        
        for item in tqdm(flattened_dataset, desc=f"Evaluating {model_name}"):
            if "text" in item:
                original = item["text"]
            else:
                print(f"Skipping item without text: {item.get('id', 'unknown')}")
                continue
            
            # Generate paraphrase
            try:
                if is_seq2seq:
                    # For BART/T5 models
                    prompt = f"paraphrase: {original}"
                    inputs = tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        max_length=1000, 
                        padding="max_length", 
                        truncation=True,
                        return_attention_mask=True
                    ).to(device)
                    
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=1000,
                        num_beams=5,
                        temperature=0.7,
                        no_repeat_ngram_size=3
                    )
                    generated_paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    # For DeepSeek and other causal models
                    user_message = f"Paraphrase the following text while preserving its meaning but changing the wording and structure: {original}"
                    
                    # Handle different models with appropriate prompt formats
                    if "deepseek" in model_name.lower() and any(x in model_name.lower() for x in ["r1", "chat"]):
                        # DeepSeek-R1/Chat specific format
                        input_text = f'<｜begin▁of▁sentence｜><｜User｜>{user_message}<｜Assistant｜><think>\nLet me analyze this text and find ways to rephrase it while keeping the same meaning.\nI need to use different vocabulary and structure.\n</think>\n\n'
                    else:
                        # Generic format for other causal LMs
                        input_text = f"User: {user_message}\n\nAssistant: <think>\nLet me analyze this text and find ways to rephrase it while keeping the same meaning.\nI need to use different vocabulary and structure.\n</think>\n\n"
                    
                    # Tokenize with attention mask explicitly
                    input_tokens = tokenizer(
                        input_text, 
                        return_tensors="pt",
                        padding=True,
                        return_attention_mask=True
                    ).to(device)
                    
                    # Generate with both input_ids and attention_mask
                    outputs = model.generate(
                        input_ids=input_tokens.input_ids,
                        attention_mask=input_tokens.attention_mask,
                        max_length=1000,
                        temperature=0.6,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id
                    )
                    
                    # Decode the full response
                    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract just the paraphrase part
                    parts = full_response.split("</think>")
                    if len(parts) > 1:
                        generated_paraphrase = parts[1].strip()
                    else:
                        # If </think> not found, try removing the prompt part
                        generated_paraphrase = full_response.replace(input_text, "").strip()
                        
                        # If the output still seems to contain the prompt, try a more aggressive approach
                        if len(generated_paraphrase) > len(input_text) * 0.9:
                            # Look for common Assistant markers
                            for marker in ["Assistant:", "AI:", "\n\n"]:
                                if marker in generated_paraphrase:
                                    generated_paraphrase = generated_paraphrase.split(marker, 1)[1].strip()
                                    break
                
                if not generated_paraphrase:
                    print(f"Warning: Empty paraphrase generated for item {item.get('id', 'unknown')}")
                
            except Exception as e:
                print(f"Error generating paraphrase for item {item.get('id', 'unknown')}: {e}")
                generated_paraphrase = ""
            
            generated_scores = self.evaluate_single(original, generated_paraphrase)
            
            # Store results
            result = {
                "id": item.get("id", f"example_{len(results)}"),
                "category": item.get("category", "unknown"),
                "original": original,
                "generated_paraphrase": generated_paraphrase,
                "scores": generated_scores
            }
            results.append(result)
        
        avg_metrics = {}
        category_metrics = {}
        
        categories = set(item.get("category", "unknown") for item in results)
        for category in categories:
            category_metrics[category] = {metric: [] for metric in self.metrics}
        
        for result in results:
            category = result.get("category", "unknown")
            for metric in self.metrics:
                score = result["scores"].get(metric, 0)
                category_metrics[category][metric].append(score)
        
        for metric in self.metrics:
            scores = [r["scores"].get(metric, 0) for r in results]
            if scores:  # Check if scores list is not empty
                avg_metrics[metric] = np.mean(scores)
            else:
                avg_metrics[metric] = 0.0
        
        category_averages = {}
        for category in category_metrics:
            category_averages[category] = {}
            for metric in self.metrics:
                scores = category_metrics[category][metric]
                if scores:  # Check if scores list is not empty
                    category_averages[category][metric] = np.mean(scores)
                else:
                    category_averages[category][metric] = 0.0
        
        aggregate = {
            "model_name": model_name,
            "dataset": dataset_path,
            "examples": len(results),
            "avg_metrics": avg_metrics,
            "category_averages": category_averages
        }
        
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump({
                        "aggregate": aggregate,
                        "detailed": results
                    }, f, indent=2)
                print(f"Results saved to {output_path}")
            except Exception as e:
                print(f"Error saving results to {output_path}: {e}")
        
        print(f"\nResults for {model_name}:")
        print("=" * 70)
        print(f"{'Metric':<20} {'Score':<10} {'Interpretation'}")
        print("-" * 70)
        
        interpretations = {
            "bertscore": {
                "excellent": 0.90,
                "good": 0.80,
                "acceptable": 0.70,
                "poor": 0.60,
                "higher_better": True,
                "description": "Semantic similarity"
            },
            "bleu_diversity": {
                "excellent": 0.80,
                "good": 0.60,
                "acceptable": 0.40,
                "poor": 0.20,
                "higher_better": True,
                "description": "Word choice diversity"
            },
            "edit_distance": {
                "excellent": 0.70,
                "good": 0.50,
                "acceptable": 0.30,
                "poor": 0.15,
                "higher_better": True,
                "description": "Character-level changes"
            },
            "syntactic_diversity": {
                "excellent": 0.70,
                "good": 0.50,
                "acceptable": 0.30,
                "poor": 0.15,
                "higher_better": True,
                "description": "Structural changes"
            },
            "harmonic_score": {
                "excellent": 0.75,
                "good": 0.60,
                "acceptable": 0.45,
                "poor": 0.30,
                "higher_better": True,
                "description": "Balance of meaning & diversity"
            }
        }
        
        for metric in self.metrics:
            score = avg_metrics[metric]
            
            interp = interpretations.get(metric, {})
            if interp:
                if interp.get("higher_better", True):
                    if score >= interp["excellent"]:
                        quality = "Excellent"
                    elif score >= interp["good"]:
                        quality = "Good"
                    elif score >= interp["acceptable"]:
                        quality = "Acceptable"
                    else:
                        quality = "Poor"
                else:
                    if score <= interp["excellent"]:
                        quality = "Excellent"
                    elif score <= interp["good"]:
                        quality = "Good"
                    elif score <= interp["acceptable"]:
                        quality = "Acceptable"
                    else:
                        quality = "Poor"
                
                interpretation = f"{quality} {interp.get('description', '')}"
            else:
                interpretation = ""
            
            print(f"{metric:<20} {score:.4f}     {interpretation}")
        
        print("\nResults by category:")
        for category in sorted(category_averages.keys()):
            print(f"\n{category.upper()} Category:")
            print("-" * 70)
            for metric in self.metrics:
                score = category_averages[category][metric]
                # Get interpretation
                interp = interpretations.get(metric, {})
                if interp:
                    if interp.get("higher_better", True):
                        if score >= interp["excellent"]:
                            quality = "Excellent"
                        elif score >= interp["good"]:
                            quality = "Good"
                        elif score >= interp["acceptable"]:
                            quality = "Acceptable"
                        else:
                            quality = "Poor"
                    else:
                        if score <= interp["excellent"]:
                            quality = "Excellent"
                        elif score <= interp["good"]:
                            quality = "Good"
                        elif score <= interp["acceptable"]:
                            quality = "Acceptable"
                        else:
                            quality = "Poor"
                    
                    interpretation = f"{quality} {interp.get('description', '')}"
                else:
                    interpretation = ""
                print(f"{metric:<20} {score:.4f}     {interpretation}")
        
        return aggregate, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate paraphrase models")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON")
    parser.add_argument("--output", type=str, default=None, help="Path to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--smoothing", type=int, default=1, choices=[0, 1, 2, 3, 4, 5, 6, 7],
                        help="BLEU smoothing method (0-7)")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token (optional)")
    parser.add_argument("--limit", type=int, help="Limit evaluation to first N examples (optional)")
    
    args = parser.parse_args()
    
    # Handle Hugging Face token
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
    elif "HUGGINGFACE_TOKEN" not in os.environ:
        # If token not provided as arg and not in environment, prompt for it
        try:
            from getpass import getpass
            print("Hugging Face token not found. Please enter your token (input will be hidden):")
            token = getpass()
            if token.strip():
                from huggingface_hub import login
                login(token=token)
                print("Successfully logged in to Hugging Face")
        except ImportError:
            print("Warning: getpass module not available. Continuing without token.")
        except Exception as e:
            print(f"Warning: Failed to login to Hugging Face: {e}")
    
    evaluator = ParaphraseEvaluator()
    
    # Set smoothing method if specified
    if args.smoothing == 0:
        evaluator.smoothing = SmoothingFunction().method0
    elif args.smoothing == 1:
        evaluator.smoothing = SmoothingFunction().method1
    elif args.smoothing == 2:
        evaluator.smoothing = SmoothingFunction().method2
    elif args.smoothing == 3:
        evaluator.smoothing = SmoothingFunction().method3
    elif args.smoothing == 4:
        evaluator.smoothing = SmoothingFunction().method4
    elif args.smoothing == 5:
        evaluator.smoothing = SmoothingFunction().method5
    elif args.smoothing == 6:
        evaluator.smoothing = SmoothingFunction().method6
    elif args.smoothing == 7:
        evaluator.smoothing = SmoothingFunction().method7
    
    try:
        evaluator.evaluate_model(args.model, args.dataset, args.output, args.device)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nError during evaluation: {e}")