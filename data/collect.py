import json
import os
import random
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import PyPDF2

nltk.download('punkt_tab')

nltk.download('punkt', quiet=False)
nltk.data.find('tokenizers/punkt')

def clean_sentence(sentence):
    """Clean and normalize a sentence"""
    # Remove extra whitespace
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    # Remove line breaks
    sentence = sentence.replace('\n', ' ')
    # Ensure sentence ends with punctuation
    if sentence and sentence[-1] not in '.?!':
        sentence += '.'
    return sentence

def extract_sentences_from_text(filepath, min_line=0, min_words=30, max_sentences=20000):
    """Extract sentences from a text file, skipping the first min_line lines"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            # Skip initial lines (preface, etc.)
            lines = file.readlines()[min_line:]
            text = ' '.join(lines)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
    
    # Extract sentences
    sentences = sent_tokenize(text)
    valid_sentences = []
    
    for sentence in sentences:
        # Clean the sentence
        clean = clean_sentence(sentence)
        
        # Check if it's a valid sentence
        if len(clean) < 10:  # Too short to be meaningful
            continue
            
        word_count = len(word_tokenize(clean))
        if word_count < min_words:  # Only keep substantial sentences
            continue
            
        if "gutenberg" in clean.lower() or "project gutenberg" in clean.lower():
            continue 
            
        valid_sentences.append(clean)
    
    # Return a random sample
    if len(valid_sentences) > max_sentences:
        return random.sample(valid_sentences, max_sentences)
    return valid_sentences

def extract_sentences_from_pdf(pdf_path, min_words=30, max_sentences=20000):
    """Extract sentences from a PDF file"""
    try:
        # Create a reader object
        reader = PyPDF2.PdfReader(pdf_path)
        
        # Extract text from each page
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return []
    
    # Extract sentences
    sentences = sent_tokenize(text)
    valid_sentences = []
    
    for sentence in sentences:
        # Clean the sentence
        clean = clean_sentence(sentence)
        
        # Check if it's a valid sentence
        if len(clean) < 10:  # Too short to be meaningful
            continue
            
        word_count = len(word_tokenize(clean))
        if word_count < min_words:  # Only keep substantial sentences
            continue
            
        valid_sentences.append(clean)
    
    # Return a random sample
    if len(valid_sentences) > max_sentences:
        return random.sample(valid_sentences, max_sentences)
    return valid_sentences

def create_dataset(target_size=500, min_words=30):
    """Create a dataset of substantial sentences"""
    collection = {
        "literary": [],
        "technical": [],
        "academic": [],
        "article": []
    }
    
    # Process text files
    print("Processing books...")
    gatsby_sentences = extract_sentences_from_text("books/gatsby.txt", min_line=1000, min_words=min_words)
    pride_sentences = extract_sentences_from_text("books/pride_and_prejudice.txt", min_line=1000, min_words=min_words)
    
    literary_sentences = gatsby_sentences + pride_sentences
    
    # Process technical files
    print("Processing technical docs...")
    react_sentences = extract_sentences_from_text("github_repo_doc/react.txt", min_line=0, min_words=min_words)
    tf_sentences = extract_sentences_from_text("github_repo_doc/tensorflow.txt", min_line=0, min_words=min_words)
    
    technical_sentences = react_sentences + tf_sentences
    
    # Process articles
    print("Processing articles...")
    article_sentences = extract_sentences_from_text("article/human_obsolete.txt", min_line=0, min_words=min_words)
    
    print("Processing academic papers...")

    academic_sentences = []
    for paper_file in ["papers/attention_is_all_you_need.txt", "papers/deepseek.txt"]:
        if os.path.exists(paper_file):
            print(f"Processing {paper_file}...")
            paper_sentences = extract_sentences_from_text(paper_file, min_line=0, min_words=min_words)
            academic_sentences.extend(paper_sentences)
    
    # Calculate sentences needed per domain
    target_per_domain = target_size // 4  # Split evenly among domains
    
    # Add sentences to collection
    for domain, sentences in [
        ("literary", literary_sentences),
        ("technical", technical_sentences),
        ("academic", academic_sentences),
        ("article", article_sentences)
    ]:
        # Sample sentences for each domain
        sample_size = min(target_per_domain, len(sentences))
        if sample_size > 0:
            sample = random.sample(sentences, sample_size)
            
            # Add to collection with IDs
            for i, text in enumerate(sample):
                collection[domain].append({
                    "id": f"{domain[:3]}_{i+1}",
                    "source": f"{domain}_source",
                    "text": text
                })
        else:
            print(f"Warning: No valid sentences found for {domain}")
    
    return collection

def main():
    print("Automatically creating a dataset of substantial sentences (30+ words)...")
    collection = create_dataset(target_size=500, min_words=30)
    
    os.makedirs("evaluation", exist_ok=True)
    
    with open("evaluation/collected_sentences.json", "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2, ensure_ascii=False)
    
    # Print stats
    total = 0
    print("\nDataset statistics:")
    
    for domain in collection:
        count = len(collection[domain])
        print(f"  {domain}: {count} sentences")
        total += count
    
    print(f"\nOVERALL TOTAL: {total} sentences")
    print("Dataset saved to 'evaluation/collected_sentences.json'")

if __name__ == "__main__":
    main()