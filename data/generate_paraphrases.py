import time
import os
from tqdm import tqdm 
from openai import OpenAI
import json

def get_api_key():
    """Prompt the user for their OpenAI API key if not set in environment"""
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("OpenAI API key not found in environment variables.")
        api_key = input("Please enter your OpenAI API key: ").strip()

        os.environ["OPENAI_API_KEY"] = api_key

    return api_key

def generate_paraphrases():
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)
    
    output_path = "evaluation/paraphrase_dataset.json"
    
    print("Loading collected sentences...")
    try:
        with open("evaluation/collected_sentences.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: collected_sentences.json not found in the evaluation directory!")
        return

    result = {}
    if os.path.exists(output_path):
        print("Found existing paraphrase dataset. Loading for updates...")
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        except json.JSONDecodeError:
            print("Error reading existing file. Starting fresh.")
            result = {}
    
    for domain in data:
        print(f"Processing {domain} domain...")
        
        if domain not in result:
            result[domain] = []
            
        # Get list of sentence IDs already processed
        processed_ids = {item["id"] for item in result[domain]}
        
        items_to_process = [item for item in data[domain] if item["id"] not in processed_ids]
        print(f"Found {len(items_to_process)} sentences to process in {domain} domain")
        
        for item in tqdm(items_to_process):
            sentence_id = item["id"]
            original_text = item["text"]

            max_retries = 3
            current_retry = 0
            success = False
            
            while current_retry < max_retries and not success:
                try:
                    # Improved prompt for more accurate paraphrases
                    system_prompt = """You are an expert at paraphrasing sentences.
                    
Your task:
1. Create EXACTLY 3 different paraphrases of the given sentence.
2. Each paraphrase MUST preserve the original meaning completely.
3. Response format must be EXACTLY 3 lines, with one paraphrase per line.
4. Do NOT add any numbers, explanations, or extra text.
5. Do NOT repeat the original sentence.
6. Each paraphrase should be distinctly different from the others.
7. Maintain the original tone, style, and complexity level.

IMPORTANT: Your entire response must contain EXACTLY 3 lines, nothing more, nothing less."""

                    response = client.chat.completions.create(
                        model="o4-mini", 
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": f'Paraphrase this sentence (Do not forget three lines with 3 different paraphrases only) : "{original_text}"'
                            }
                        ],
                        max_completion_tokens=1000,
                    )

                    paraphrase_text = response.choices[0].message.content.strip()
                    paraphrases = [p.strip() for p in paraphrase_text.split('\n') if p.strip()]

                    # Check if we got exactly 3 paraphrases
                    if len(paraphrases) == 3:
                        success = True
                    else:
                        print(f"\nWarning: Got {len(paraphrases)} paraphrases instead of 3 for {sentence_id}. Retry {current_retry + 1}/{max_retries}")
                        current_retry += 1
                        time.sleep(1)  # Wait a bit before retrying

                except Exception as e:
                    print(f"\nError processing {sentence_id} (try {current_retry + 1}/{max_retries}): {str(e)}")
                    current_retry += 1
                    time.sleep(2)  # Wait longer after an error
                    
                    if "auth" in str(e).lower() or "api key" in str(e).lower():
                        print("API key issue detected. Please check your API key.")
                        new_key = input("Enter a new OpenAI API key (or press Enter to skip): ").strip()
                        if new_key:
                            os.environ["OPENAI_API_KEY"] = new_key
                            client = OpenAI(api_key=new_key)
            
            if success:
                new_item = item.copy()
                new_item["paraphrases"] = paraphrases
                result[domain].append(new_item)
                
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                print(f"\nFailed to generate 3 paraphrases for {sentence_id} after {max_retries} attempts. Skipping for now.")
                
            time.sleep(0.5)

    total_processed = sum(len(result[domain]) for domain in result)
    total_in_source = sum(len(data[domain]) for domain in data)
    missing_count = total_in_source - total_processed
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {total_processed}/{total_in_source} sentences")
    if missing_count > 0:
        print(f"Remaining sentences to process: {missing_count}")
        print("Run the script again to retry these sentences.")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    generate_paraphrases()