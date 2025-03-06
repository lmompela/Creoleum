def read_file(cleaned_file):
    with open(cleaned_file, 'r', encoding='utf-8') as file:
        dataset = file.readlines()
    file.close()
    return dataset

def find_token(token, dataset):
    data = read_file(dataset)
    sentences = []
    cnt = 1 
    for line in data:
        line = line.split()
        for word in line:
            if token == word:
                sentences.append(line)
    return sentences

def write_to_file(sentences, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            #print(sentence)
            f.write(sentence[0] + "\t"+ sentence[1] + "\t" + sentence[2] + "\t" + sentence[3] + "\t" + ' '.join(sentence[4:]) + '\n')
    f.close()

def main():
    output_file = './sentences_with_eti.csv'
    input_file = 'datasets/cleaned_dataset_mar_full.txt'
    token = ["eti", "éti"]
    for t in token:
        write_to_file(find_token(t, input_file), output_file)

main()