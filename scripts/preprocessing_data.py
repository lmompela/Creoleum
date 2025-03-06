import os 
import re

def extract_data(input_folder, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        cnt = 1
        for file_name in os.listdir(input_folder):
            with open(os.path.join(input_folder, file_name), 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                pattern = r'\b[A-Z][a-z]+, \d{1,2} [A-Z][a-z]+, \d{4} - \d{2}:\d{2}\b'
                match = re.find(pattern, content)
                content = f.readlines()
                for line in content:
                    #file.write(str(cnt) + "\t" + content + "\n")
                    file.write("\t".join(file_name, "_".join(match.split()), line) + "\n")
                    cnt += 1

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

