# DeepSeek Paraphrase

Fine-tuned paraphrasing model using DeepSeek-R1-Distill-Qwen-7B with LoRA adaptation. Outperforms baseline models like BART and T5 in generating high-quality paraphrases that balance semantic preservation with lexical and structural diversity.

## Features

- Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- Reasoning prompts for better paraphrase generation
- Multi-domain evaluation (literary, technical, academic, article)
- Evaluation metrics (BERTScore, BLEU diversity, edit distance, etc.)

## Project Structure

```
./
├── data/                            # Data directory
│   ├── article/
│   │   └── human_obsolete.txt
│   ├── books/
│   │   ├── gatsby.txt
│   │   └── pride_and_prejudice.txt
│   ├── deepseek/
│   │   ├── deepseek_train.json      # Training data in SFT format
│   │   └── deepseek_val.json        # Validation data in SFT format
│   ├── evaluation/                  # Evaluation datasets and results
│   │   ├── splits/
│   │   ├── collected_sentences.json # Extracted sentences from source texts
│   │   ├── paraphrase_dataset.json  # Original texts with gold paraphrases
│   │   ├── results_bart.json        # BART model evaluation results
│   │   ├── results_deepseek.json    # DeepSeek model evaluation results
│   │   └── results_t5.json          # T5 model evaluation results
│   ├── github_repo_doc/
│   │   ├── react.txt
│   │   └── tensorflow.txt
│   ├── papers/
│   │   ├── attention_is_all_you_need.txt
│   │   └── deepseek.txt
│   ├── collect.py                   # Dataset creation script
│   ├── create_deepseek_dataset.py   # Prepares data for DeepSeek fine-tuning
│   ├── generate_paraphrases.py      # Creates gold paraphrases with OpenAI API
│   └── split.py                     # Dataset splitting script
├── deepseek-paraphrase-lora/        # Trained model checkpoints this dir is not pushed (too big)
│   ├── checkpoint-48
│   ├── checkpoint-432
│   ├── checkpoint-480
│   ├── final/                       # Final model checkpoint
│   └── runs/                        # Training logs
├── evaluate.py                      # Model evaluation script
├── README.md                        # This file
├── requirements.txt                 # Dependencies
├── run.ipynb                        # End-to-end notebook
├── sft_format_example.json          # Example of SFT training format
├── test.py                          # Testing utilities
└── train.py                         # Model fine-tuning script
```

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Usage

1. **Data Collection and Preparation**:
   ```bash
   python data/collect.py                   # Extract sentences from source texts
   python data/generate_paraphrases.py      # Generate gold paraphrases
   python data/split.py                     # Split into train/test
   python data/create_deepseek_dataset.py   # Format for DeepSeek training
   ```

2. **Model Training**:
   ```bash
   python train.py
   ```

3. **Model Evaluation**:
   ```bash
   python evaluate.py --model deepseek-paraphrase-lora/final --dataset data/evaluation/collected_sentences.json --output data/evaluation/results_deepseek.json
   ```

4. **End-to-End Notebook**:
   - The `run.ipynb` notebook contains a complete walkthrough of the entire process

## Results

The DeepSeek-LoRA model achieves:
Metric               Score      Interpretation
----------------------------------------------------------------------
bertscore            0.9521     Excellent Semantic similarity
bleu_diversity       0.5135     Acceptable Word choice diversity
edit_distance        0.3439     Acceptable Character-level changes
syntactic_diversity  0.1472     okay Structural changes
harmonic_score       0.4684     Acceptable Balance of meaning & diversity

## Example Paraphrases

### Literary Text
**Original**: "Day after day passed away without bringing any other tidings of him than the report which shortly prevailed in Meryton of his coming no more to Netherfield the whole winter; a report which highly incensed Mrs. Bennet, and which she never failed to contradict as a most scandalous falsehood."

**DeepSeek-LoRA**: "Day after day passed without any news of him other than the report in Meryton that he would not return to Netherfield for the entire winter; a report that highly upset Mrs. Bennet and which she never failed to deny as a scandalous falsehood."

### Technical Text
**Original**: "Learn Once, Write Anywhere: We don't make assumptions about the rest of your technology stack, so you can develop new features in React without rewriting existing code."

**DeepSeek-LoRA**: "We don't make assumptions about the rest of your technology stack, so you can develop new features in React without rewriting existing code."