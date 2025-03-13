#!/usr/bin/env python3
import os
import re
import pandas as pd
import unicodedata
import argparse
import random
import math
from collections import defaultdict, Counter
import nltk

# Download NLTK tokenizer data if not already available
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

##############################
# Helper functions for training n-gram models using balanced datasets
##############################
def balance_datasets_by_tokens(creole_file, french_file):
    """
    Read both training datasets as text, tokenize them, and then sample from the larger (French)
    dataset so that both have roughly the same total token count.
    """
    with open(creole_file, 'r', encoding='utf-8') as f:
        creole_text = f.read()
    with open(french_file, 'r', encoding='utf-8') as f:
        french_text = f.read()
    
    # Tokenize by splitting on whitespace
    creole_tokens = re.findall(r'\S+', creole_text)
    french_tokens = re.findall(r'\S+', french_text)
    
    print(f"Total Creole tokens: {len(creole_tokens)}")
    print(f"Total French tokens: {len(french_tokens)}")
    
    # Use the total token count from Creole as target
    target = len(creole_tokens)
    random.shuffle(french_tokens)
    balanced_french_tokens = french_tokens[:target]
    
    balanced_creole_text = ' '.join(creole_tokens)
    balanced_french_text = ' '.join(balanced_french_tokens)
    return balanced_creole_text, balanced_french_text

def train_ngram_model(text, n=3):
    """
    Train a character-level n-gram model (default trigram) from the given text.
    Returns a dictionary mapping n-grams to a Counter of following characters (normalized to probabilities).
    """
    model = defaultdict(Counter)
    text = text.replace("\n", " ")
    for i in range(len(text) - n):
        gram = text[i:i+n]
        next_char = text[i+n]
        model[gram][next_char] += 1
    # Normalize counts to probabilities
    for gram, counter in model.items():
        total = float(sum(counter.values()))
        for char in counter:
            counter[char] /= total
    return model

def score_text(text, model, n=3, epsilon=1e-6):
    """
    Compute the average log-likelihood of the text under the given n-gram model.
    This average (total log-likelihood divided by number of n-grams) normalizes for text length.
    """
    text = text.replace("\n", " ")
    total_log = 0.0
    count = 0
    for i in range(len(text) - n):
        gram = text[i:i+n]
        next_char = text[i+n]
        prob = model[gram].get(next_char, epsilon)
        total_log += math.log(prob)
        count += 1
    return total_log / count if count > 0 else float('-inf')

def softmax(scores):
    """
    Convert a list of log-likelihood scores to normalized probabilities.
    """
    max_score = max(scores)
    exps = [math.exp(s - max_score) for s in scores]
    sum_exps = sum(exps)
    return [exp_val / sum_exps for exp_val in exps]

def identify_language(sentence, creole_model, french_model, n=3):
    """
    Compute overall language proportions for the sentence using the two models.
    Returns a dictionary with keys 'Creole' and 'French'.
    """
    score_creole = score_text(sentence.lower(), creole_model, n=n)
    score_french = score_text(sentence.lower(), french_model, n=n)
    probs = softmax([score_creole, score_french])
    return {'Creole': probs[0], 'French': probs[1]}

def classify_tokens(sentence, creole_model, french_model, window_size=3, n=3):
    """
    For debugging: Classify each token in the sentence using a sliding window approach.
    Each token appears in overlapping windows; the final probability is the average over those windows.
    Returns a list of dictionaries with token-level probabilities and a predicted label.
    """
    tokens = word_tokenize(sentence)
    token_probs = [ [] for _ in tokens ]
    num_tokens = len(tokens)
    
    for start in range(num_tokens):
        end = start + window_size
        if end > num_tokens:
            end = num_tokens
            start = max(0, end - window_size)
        window_tokens = tokens[start:end]
        window_text = ' '.join(window_tokens)
        proportions = identify_language(window_text, creole_model, french_model, n=n)
        for idx in range(start, end):
            token_probs[idx].append(proportions)
    
    averaged_results = []
    for token, probs_list in zip(tokens, token_probs):
        avg_creole = sum(p['Creole'] for p in probs_list) / len(probs_list)
        avg_french = sum(p['French'] for p in probs_list) / len(probs_list)
        averaged_results.append({
            'token': token,
            'Creole': avg_creole,
            'French': avg_french,
            'label': 'Creole' if avg_creole > avg_french else 'French'
        })
    return averaged_results

##############################
# Helper function for sentence splitting
##############################
def split_sentences(text, language='french'):
    """
    Split text into sentences using both newline characters and nltk.sent_tokenize.
    This way, isolated lines (titles, subtitles, etc.) are treated as separate sentences.
    """
    # First, split on one or more newline characters.
    parts = re.split(r'\n+', text)
    sentences = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Further split the part using nltk.sent_tokenize in case it contains multiple sentences.
        sentences.extend(sent_tokenize(part, language=language))
    return sentences

##############################
# Functions for processing dataset files
##############################
def normalize_text(text):
    """Normalize text to standardize punctuation and encoding."""
    return unicodedata.normalize('NFKC', text)

def extract_date_martinican(text):
    """
    Extract date from Martinican Creole files.
    Expected pattern example: "Lundi, 23 Août, 2021"
    """
    match = re.search(r'([A-Za-zÀ-ÿ]+,\s*\d{1,2}\s*[A-Za-zÀ-ÿ]+,\s*\d{4})', text)
    if match:
        return match.group(1)
    return ""

def process_martinican_file(file_path, creole_model, french_model, n=3):
    """
    Process a single Martinican Creole text file using the new language model.
    Uses split_sentences() to treat newline as an end-of-sentence.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin1') as f:
            content = f.read()
    content = normalize_text(content)
    date = extract_date_martinican(content)
    author = os.path.basename(os.path.dirname(file_path))
    sentences = split_sentences(content, language='french')
    rows = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            proportions = identify_language(sentence, creole_model, french_model, n=n)
            predicted_lang = 'Creole' if proportions['Creole'] > proportions['French'] else 'French'
            # Debug: Print token-level classifications
            token_classifications = classify_tokens(sentence, creole_model, french_model, window_size=3, n=n)
            print("Token-level classifications for sentence:")
            for entry in token_classifications:
                print(f"  Token: {entry['token']}, Creole: {entry['Creole']:.3f}, French: {entry['French']:.3f}, Label: {entry['label']}")
            rows.append({
                "Language": predicted_lang,
                "Author": author,
                "Date": date,
                "Content": sentence,
                "Creole Score": proportions['Creole'],
                "French Score": proportions['French']
            })
    return rows

def process_antilla_file(file_path, creole_model, french_model, n=3):
    """
    Process the single Antilla TXT file with multiple articles using the new language model.
    Splits the text into articles based on the "Kréyolad" marker and uses a modified regex to extract the date.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin1') as f:
            content = f.read()
    content = normalize_text(content)
    rows = []
    articles = re.split(r'(?=Kréyolad\s+\d+)', content)
    for article in articles:
        article = article.strip()
        if not article:
            continue
        # Use MULTILINE to match the date on a line starting with "Antilla"
        match = re.search(r'^Antilla\s+\d+,\s*(.+)$', article, re.MULTILINE)
        date = match.group(1).strip() if match else ""
        author = "Antilla"
        sentences = split_sentences(article, language='french')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                proportions = identify_language(sentence, creole_model, french_model, n=n)
                predicted_lang = 'Creole' if proportions['Creole'] > proportions['French'] else 'French'
                token_classifications = classify_tokens(sentence, creole_model, french_model, window_size=3, n=n)
                print("Token-level classifications for sentence:")
                for entry in token_classifications:
                    print(f"  Token: {entry['token']}, Creole: {entry['Creole']:.3f}, French: {entry['French']:.3f}, Label: {entry['label']}")
                rows.append({
                    "Language": predicted_lang,
                    "Author": author,
                    "Date": date,
                    "Content": sentence,
                    "Creole Score": proportions['Creole'],
                    "French Score": proportions['French']
                })
    return rows

##############################
# Main function
##############################
def main():
    parser = argparse.ArgumentParser(
        description="Process Martinican Creole and Antilla datasets into a CSV file using a custom n-gram language classifier."
    )
    parser.add_argument('--martinican_dir', type=str, help="Path to the Martinican Creole dataset directory (with subfolders by author)")
    parser.add_argument('--antilla_file', type=str, help="Path to the Antilla dataset TXT file")
    parser.add_argument('--output_csv', type=str, default="processed_dataset.csv", help="Output CSV file path")
    # Paths for external training datasets (balanced)
    parser.add_argument('--creole_training', type=str, default="datasets/creole_dataset.txt", help="Path to external Creole training dataset")
    parser.add_argument('--french_training', type=str, default="datasets/french_dataset.txt", help="Path to external French training dataset")
    args = parser.parse_args()
    
    # Balance the external training datasets by token count and train n-gram models
    n = 3  # Using trigrams; adjust as needed
    balanced_creole_text, balanced_french_text = balance_datasets_by_tokens(args.creole_training, args.french_training)
    creole_model = train_ngram_model(balanced_creole_text, n=n)
    french_model = train_ngram_model(balanced_french_text, n=n)
    
    all_rows = []
    # Process Martinican files if directory is provided.
    if args.martinican_dir:
        for root, _, files in os.walk(args.martinican_dir):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    rows = process_martinican_file(file_path, creole_model, french_model, n=n)
                    all_rows.extend(rows)
    # Process Antilla file if provided.
    if args.antilla_file:
        rows = process_antilla_file(args.antilla_file, creole_model, french_model, n=n)
        all_rows.extend(rows)
    
    if all_rows:
        df = pd.DataFrame(all_rows, columns=["Language", "Author", "Date", "Content", "Creole Score", "French Score"])
        df.to_csv(args.output_csv, index=False, encoding='utf-8')
        print(f"\nCSV file saved to {args.output_new_csv}")
    else:
        print("No data processed.")

if __name__ == '__main__':
    main()
