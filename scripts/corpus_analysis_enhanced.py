import pandas as pd
import nltk
import argparse
import re
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# --- Tokenization functions ---
def custom_tokenize(text):
    """
    Tokenize text while removing punctuation.
    This pattern keeps only letters (including accented letters).
    """
    tokenizer = RegexpTokenizer(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
    return tokenizer.tokenize(text)

def tokenize_lower(text):
    """
    Tokenize and convert all tokens to lower case.
    """
    tokens = custom_tokenize(text)
    return [t.lower() for t in tokens]

# --- Targeted n-gram and collocation analysis functions ---
def targeted_ngram_analysis(texts, target, n=2, top_n=10):
    """
    Computes frequency of n-grams (e.g., bigrams/trigrams) that include the target word.
    Only tokens matching our regex (i.e. words without punctuation) are considered.
    """
    target = target.lower()
    counter = Counter()
    for text in texts:
        tokens = tokenize_lower(text)
        for gram in ngrams(tokens, n):
            if target in gram:
                counter[gram] += 1
    return counter.most_common(top_n)

def targeted_collocation_analysis(texts, target, min_freq=3, top_n=10):
    """
    Computes collocation (bigram) statistics using PMI for bigrams that include the target word.
    """
    target = target.lower()
    tokens_all = []
    for text in texts:
        tokens_all.extend(tokenize_lower(text))
    finder = BigramCollocationFinder.from_words(tokens_all)
    finder.apply_freq_filter(min_freq)
    bigram_measures = BigramAssocMeasures()
    collocs = []
    for bigram, freq in finder.ngram_fd.items():
        if target in bigram:
            score = finder.score_ngram(bigram_measures.pmi, bigram[0], bigram[1])
            collocs.append((bigram, score, freq))
    collocs_sorted = sorted(collocs, key=lambda x: x[1], reverse=True)
    return collocs_sorted[:top_n]

# --- KWIC extraction functions ---
def extract_kwic(text, target):
    """
    Extracts the first occurrence of the target word from the text (case-insensitive),
    and returns a tuple: (before, target, after). If not found, returns None.
    """
    pattern = re.compile(r"(.*?)(\b" + re.escape(target) + r"\b)(.*)", flags=re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if match:
        before = match.group(1).strip()
        word = match.group(2).strip()  # preserves the original casing
        after = match.group(3).strip()
        return before, word, after
    else:
        return None

def extract_kwic_for_targets(texts, targets):
    """
    For each text in texts, check for each target word (case-insensitive) and,
    if found, extract its KWIC (only the first occurrence per target per text).
    Returns a list of dictionaries with keys: target_word, context_before, target, context_after.
    """
    entries = []
    for text in texts:
        for target in targets:
            res = extract_kwic(text, target)
            if res:
                before, word, after = res
                entries.append({
                    "target_word": target,
                    "context_before": before,
                    "target": word,
                    "context_after": after
                })
    return entries

# --- Clustering function for KWIC entries ---
def cluster_kwic(kwic_entries, num_clusters):
    """
    Computes sentence embeddings for each KWIC occurrence (concatenating before, target, after)
    and clusters them using KMeans.
    Returns the list of KWIC entries with an added "cluster" key.
    """
    sentences = [
        entry["context_before"] + " " + entry["target"] + " " + entry["context_after"]
        for entry in kwic_entries
    ]
    if not sentences:
        print("No KWIC entries to cluster.")
        return None
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    for entry, cluster in zip(kwic_entries, clusters):
        entry["cluster"] = int(cluster)
    return kwic_entries

# --- Main processing function ---
def main():
    parser = argparse.ArgumentParser(
        description="Detailed corpus analysis comparing target words (e.g., 'éti' and 'ki')."
    )
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("--text_column", default="text", help="Name of the column containing text data")
    parser.add_argument("--target_words", default="éti,ki",
                        help="Comma-separated target words (e.g., 'éti,ki')")
    parser.add_argument("--mode", choices=["analysis", "kwic", "cluster", "all"], default="all",
                        help="Operation mode: analysis (n-gram/collocation), kwic (KWIC extraction), cluster (KWIC clustering), or all")
    parser.add_argument("--ngram_n", type=int, default=2, help="n for n-gram analysis (e.g., 2 for bigrams, 3 for trigrams)")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top n-grams/collocations to display")
    parser.add_argument("--min_freq", type=int, default=3, help="Minimum frequency for collocation analysis")
    parser.add_argument("--cluster_num", type=int, default=3, help="Number of clusters for KWIC clustering")
    parser.add_argument("--kwic_output", default="kwic_extracted.csv", help="Output CSV file for KWIC extraction")
    args = parser.parse_args()

    # Read the input CSV file
    df = pd.read_csv(args.input_csv)
    texts = df[args.text_column].dropna().tolist()

    # Download necessary NLTK data (if not already installed)
    nltk.download('punkt', quiet=True)

    # Prepare target words list (strip spaces and lower-case for internal processing)
    targets = [t.strip() for t in args.target_words.split(",")]

    # --- Analysis Mode: targeted n-gram and collocation analysis ---
    if args.mode in ["analysis", "all"]:
        for target in targets:
            print(f"\n=== Analysis for target word: '{target}' ===")
            print(f"\nTargeted {args.ngram_n}-gram Analysis (only n-grams containing '{target}'):")
            ngram_results = targeted_ngram_analysis(texts, target, n=args.ngram_n, top_n=args.top_n)
            for gram, count in ngram_results:
                print(f"{gram}: {count}")

            print(f"\nTargeted Collocation Analysis (Bigrams containing '{target}'):")
            colloc_results = targeted_collocation_analysis(texts, target, min_freq=args.min_freq, top_n=args.top_n)
            for bigram, score, freq in colloc_results:
                print(f"{bigram}: PMI score = {score:.3f}, Frequency = {freq}")

    # --- KWIC Extraction Mode: transform CSV to KWIC CSV ---
    if args.mode in ["kwic", "all"]:
        kwic_entries = extract_kwic_for_targets(texts, targets)
        if kwic_entries:
            kwic_df = pd.DataFrame(kwic_entries)
            kwic_df.to_csv(args.kwic_output, index=False)
            print(f"\nKWIC extraction saved to {args.kwic_output}")
        else:
            print("\nNo KWIC entries found for the target words.")

    # --- Cluster Analysis Mode: perform clustering on KWIC entries and output CSV files ---
    if args.mode in ["cluster", "all"]:
        # First, extract KWIC entries (if not already done)
        kwic_entries = extract_kwic_for_targets(texts, targets)
        if not kwic_entries:
            print("\nNo KWIC entries found for clustering.")
        else:
            # Process each target word separately
            for target in targets:
                target_kwic = [entry for entry in kwic_entries if entry["target_word"].lower() == target.lower()]
                if not target_kwic:
                    print(f"\nNo KWIC entries found for target word '{target}' for clustering.")
                    continue
                clustered = cluster_kwic(target_kwic, num_clusters=args.cluster_num)
                if clustered is not None:
                    cluster_df = pd.DataFrame(clustered)
                    # Reorder columns: cluster, context_before, target, context_after
                    cluster_df = cluster_df[["cluster", "context_before", "target", "context_after"]]
                    output_file = f"clusters_{target}.csv"
                    cluster_df.to_csv(output_file, index=False)
                    print(f"\nCluster analysis for '{target}' saved to {output_file}")

if __name__ == "__main__":
    main()
