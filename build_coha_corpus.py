import json
import nltk
import os
import argparse


sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def build_train_test(input_folders, lm_output_train, lm_output_test):
    output_train = open(lm_output_train, 'w', encoding='utf')
    output_test = open(lm_output_test, 'w', encoding='utf')
    all_sents = 0
    for input_folder in input_folders:
        num_docs = len(os.listdir(input_folder))
        limit = int(0.9 * num_docs)
        counter = 0
        for file in os.listdir(input_folder):

            path = os.path.join(input_folder, file)

            with open(path, 'r', encoding='utf8') as f:
                text = f.read()
                text = text.replace('<p>', '')
                text = text.replace('<P>', '')
                text = text.replace('@', '')
                text = text.replace('"', '')
                sents = sent_tokenizer.tokenize(text)
                counter += 1
                if counter % 100 == 0:
                    print('Num docs: ', counter)

                for sent in sents:
                    sent = " ".join(sent.split())
                    sent = sent.strip() + '\n'
                    if len(sent.strip()) > 2:
                        if counter < limit:
                            output_train.write(sent)
                        else:
                            output_test.write(sent)
                all_sents += counter

    output_train.close()
    output_test.close()
    print('LM train and test sets created.')



def build_data_sets(input_folders, output_folders):

    for input_folder, output_folder in zip(input_folders, output_folders):
        output = open(output_folder, 'w', encoding='utf')
        for file in os.listdir(input_folder):
            path = os.path.join(input_folder, file)
            with open(path, 'r', encoding='utf8') as f:
                text = f.read()
                text = text.replace('<p>', '')
                text = text.replace('<P>', '')
                text = text.replace('@', '')
                text = text.replace('"', '')
                text = " ". join(text.split())
                text = {'content': text, 'title': ''}
                line = json.dumps(text) + '\n'
                output.write(line)
        output.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folders', type=str,
                        help='Path to COHA folders containing articles for each temporal slice separated by ";".',
                        default='data/coha/COHA_1960;data/coha/COHA_1990')
    parser.add_argument('--output_files', type=str, help='Path to output files containing text for each'
                                                           'temporal slice separated by ";". Should correspond'
                                                           'to the number and order of input folders.',
                        default='data/coha/coha_1960.txt;data/coha/coha_1990.txt')
    parser.add_argument('--lm_output_train', type=str,
                        help='Path to output .txt file used for language model training',
                        default='data/coha/train.txt')
    parser.add_argument('--lm_output_test', type=str,
                        help='Path to output .txt file used for language model validation',
                        default='data/coha/test.txt')
    args = parser.parse_args()

    input_folders = args.input_folders.split(';')
    output_files = args.output_files.split(';')

    build_train_test(input_folders, args.lm_output_train, args.lm_output_test)
    build_data_sets(input_folders, output_files)






