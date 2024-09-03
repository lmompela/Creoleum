import re
import difflib

def remove_non_french_chars(text):
    # Replace curly apostrophes with straight apostrophes
    text = text.replace("’", "'")
    
    # Define a regex pattern to match non-French characters
    # This pattern keeps letters, digits, common punctuation, and French diacritics
    pattern = re.compile(r'[^a-zA-Z0-9À-ÖØ-öø-ÿ\s.,;:!?\'"-]')
    # Replace non-French characters with an empty string
    cleaned_text = pattern.sub('', text)
    return cleaned_text

def clean_dataset(input_file):
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()

    cleaned_content = remove_non_french_chars(content)
    return cleaned_content

def compare_datasets(original_file, cleaned_content):
    with open(original_file, 'r', encoding='utf-8', errors='ignore') as file:
        original_content = file.read()
    
    # Generate a unified diff
    diff = difflib.unified_diff(
        original_content.splitlines(keepends=True),
        cleaned_content.splitlines(keepends=True),
        fromfile='original',
        tofile='cleaned'
    )
    
    # Print the diff to show differences
    for line in diff:
        print(line, end='')

if __name__ == "__main__":
    input_file = 'datasets/dataset_mar_full.txt'  # Replace with your input file path
    cleaned_file = 'datasets/cleaned_dataset_mar_full.txt'  # Replace with your desired output file path

    # Clean the dataset
    cleaned_content = clean_dataset(input_file)

    # Save the cleaned content to a file
    with open(cleaned_file, 'w', encoding='utf-8') as file:
        file.write(cleaned_content)

    # Compare the original and cleaned datasets
    compare_datasets(input_file, cleaned_content)
    print(f"Cleaned dataset saved to {cleaned_file}")
