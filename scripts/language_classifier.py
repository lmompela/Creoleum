import random
import math
from collections import defaultdict, Counter
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

def balance_datasets_by_tokens(creole_file, french_file):
    """
    Read both datasets as text, tokenize them, and then sample from the larger (French) dataset 
    so that both have roughly the same total token count.
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
    # Randomly shuffle French tokens and select only as many as in the Creole dataset
    random.shuffle(french_tokens)
    balanced_french_tokens = french_tokens[:target]
    
    # Reconstruct balanced texts
    balanced_creole_text = ' '.join(creole_tokens)
    balanced_french_text = ' '.join(balanced_french_tokens)
    return balanced_creole_text, balanced_french_text

def train_ngram_model(text, n=3):
    """
    Train a character-level n-gram (default trigrams) model.
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
    This normalization (dividing by number of n-grams) makes scores comparable across texts.
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
    Compute overall language proportions for the sentence using average log-likelihood scores.
    """
    score_creole = score_text(sentence.lower(), creole_model, n=n)
    score_french = score_text(sentence.lower(), french_model, n=n)
    probs = softmax([score_creole, score_french])
    return {'Creole': probs[0], 'French': probs[1]}

def classify_tokens(sentence, creole_model, french_model, window_size=3, n=3):
    """
    Classify each token by computing language probabilities over a sliding window.
    Each token will appear in multiple windows; the final probability is the average over all windows.
    Returns a list of dictionaries, one per token, with its probabilities and assigned label.
    """
    tokens = word_tokenize(sentence)
    token_probs = [ [] for _ in tokens ]
    num_tokens = len(tokens)
    
    # Slide a window over the tokens
    for start in range(num_tokens):
        end = start + window_size
        if end > num_tokens:
            end = num_tokens
            start = max(0, end - window_size)
        window_tokens = tokens[start:end]
        window_text = ' '.join(window_tokens)
        proportions = identify_language(window_text, creole_model, french_model, n=n)
        # Assign the window's probabilities to all tokens covered by it
        for idx in range(start, end):
            token_probs[idx].append(proportions)
    
    # Average probabilities for each token
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

def main():
    # Define file paths for your datasets
    creole_file = 'datasets/creole_dataset.txt'
    french_file = 'datasets/french_dataset.txt'
    
    # Balance the datasets by total token count
    balanced_creole_text, balanced_french_text = balance_datasets_by_tokens(creole_file, french_file)
    
    # Train n-gram models (using trigrams in this example)
    n = 3
    creole_model = train_ngram_model(balanced_creole_text, n=n)
    french_model = train_ngram_model(balanced_french_text, n=n)
    
    # Test sentence for overall language identification
    test_sentence = "Mwen ka chanté et j'aime la musique"
    overall = identify_language(test_sentence, creole_model, french_model, n=n)
    print("Overall language proportions:", overall)
    
    # Token-level classification using sliding window segmentation
    token_classifications = classify_tokens(test_sentence, creole_model, french_model, window_size=3, n=n)
    print("\nToken-level classifications:")
    for entry in token_classifications:
        print(f"Token: {entry['token']}, Creole: {entry['Creole']:.3f}, French: {entry['French']:.3f}, Label: {entry['label']}")

if __name__ == '__main__':
    main()
