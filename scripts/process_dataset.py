#!/usr/bin/env python3
import os
import re
import pandas as pd
import unicodedata
import argparse
from langdetect import detect_langs  # pip install langdetect
import nltk

# Ensure sentence tokenizer models are available
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def normalize_text(text):
    """Normalize text to standardize punctuation and encoding."""
    return unicodedata.normalize('NFKC', text)

def detect_language(sentence):
    """
    Detect language and return a tuple (language, confidence).
    Here we check if French is detected; otherwise, we label as Creole.
    """
    try:
        langs = detect_langs(sentence)
        # Look for French; if found above a certain threshold, mark as French.
        for lang in langs:
            if lang.lang == 'fr':
                return ("French", lang.prob)
        # Fallback: if French isn’t dominant, assume Creole.
        return ("Creole", langs[0].prob)
    except Exception:
        return ("Unknown", 0)

def extract_date_martinican(text):
    """
    Extract date from Martinican Creole files.
    Expected pattern example: "Lundi, 23 Août, 2021"
    """
    match = re.search(r'([A-Za-zÀ-ÿ]+,\s*\d{1,2}\s*[A-Za-zÀ-ÿ]+,\s*\d{4})', text)
    if match:
        return match.group(1)
    return ""

def process_martinican_file(file_path):
    """Process a single Martinican Creole text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin1') as f:
            content = f.read()
    content = normalize_text(content)
    date = extract_date_martinican(content)
    # Author is taken from the directory name (assuming structure: martinican_creole_original/author/filename.txt)
    author = os.path.basename(os.path.dirname(file_path))
    sentences = sent_tokenize(content, language='french')  # using French tokenizer as a starting point
    rows = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            lang, conf = detect_language(sentence)
            rows.append({
                "Language": lang,
                "Author": author,
                "Date": date,
                "Content": sentence,
                "Confidence Score": conf
            })
    return rows

def process_antilla_file(file_path):
    """Process the single Antilla TXT file with multiple articles."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin1') as f:
            content = f.read()
    content = normalize_text(content)
    rows = []
    # Split into articles using the marker "Kréyolad" followed by a number.
    articles = re.split(r'(?=Kréyolad\s+\d+)', content)
    for article in articles:
        article = article.strip()
        if not article:
            continue
        # Extract the date using a pattern like: "Antilla <number>, <date>"
        match = re.search(r'Antilla\s+\d+,\s+([^\n]+)', article)
        date = match.group(1).strip() if match else ""
        author = "Antilla"
        # Segment article into sentences
        sentences = sent_tokenize(article, language='french')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                lang, conf = detect_language(sentence)
                rows.append({
                    "Language": lang,
                    "Author": author,
                    "Date": date,
                    "Content": sentence,
                    "Confidence Score": conf
                })
    return rows

def main():
    parser = argparse.ArgumentParser(description="Process Martinican Creole and Antilla datasets into a CSV file.")
    parser.add_argument('--martinican_dir', type=str, help="Path to the Martinican Creole dataset directory (with subfolders by author)")
    parser.add_argument('--antilla_file', type=str, help="Path to the Antilla dataset TXT file")
    parser.add_argument('--output_csv', type=str, default="processed_dataset.csv", help="Output CSV file path")
    args = parser.parse_args()
    
    all_rows = []
    # Process Martinican files if a directory is provided.
    if args.martinican_dir:
        for root, _, files in os.walk(args.martinican_dir):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    rows = process_martinican_file(file_path)
                    all_rows.extend(rows)
    # Process the Antilla file if provided.
    if args.antilla_file:
        rows = process_antilla_file(args.antilla_file)
        all_rows.extend(rows)
    
    if all_rows:
        df = pd.DataFrame(all_rows, columns=["Language", "Author", "Date", "Content", "Confidence Score"])
        df.to_csv(args.output_csv, index=False, encoding='utf-8')
        print(f"CSV file saved to {args.output_csv}")
    else:
        print("No data processed.")

if __name__ == '__main__':
    main()
