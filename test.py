import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_deepseek_thinking_format():
    print("Loading DeepSeek tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        # For quick testing, you could use the smaller model
        # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Using fallback tokenizer for demonstration only")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Test text
    original_text = "The quick brown fox jumps over the lazy dog."
    user_message = f"Paraphrase the following text: {original_text}"
    
    # Test different formats
    print("\n\n=== FORMAT 1: User message with chat template ===")
    format1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False
    )
    print(format1)
    
    print("\n\n=== FORMAT 2: User message + Assistant with chat template ===")
    format2 = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ""}  # Empty assistant message
        ],
        tokenize=False
    )
    print(format2)
    
    print("\n\n=== FORMAT 3: User message + Assistant with manually added thinking token ===")
    format3 = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False
    ) + "<｜Assistant｜>" + "<think>\n"
    print(format3)
    
    print("\n\n=== FORMAT 4: User message + Assistant with thinking content in the message ===")
    format4 = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "<think>\nLet me analyze this text.\n</think>\n\nParaphrase here."}
        ],
        tokenize=False
    )
    print(format4)
    
    # Check if the model is available for quick testing
    try:
        print("\n\nAttempting to load small model for quick generation test...")
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        
        # Test Generation with enforced thinking
        print("\n\n=== TESTING GENERATION WITH ENFORCED THINKING ===")
        formatted_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False
        ) + "<think>\n"
        
        input_ids = tokenizer(formatted_input, return_tensors="pt").input_ids
        
        # Generate only a short continuation to see if thinking is maintained
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=30,
                do_sample=False
            )
        
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        print(f"Input + Generated:\n{output_text}")
        
    except Exception as e:
        print(f"Error during model testing: {e}")
        print("Skipping generation test - please run on machine with GPU and model access")

if __name__ == "__main__":
    test_deepseek_thinking_format()