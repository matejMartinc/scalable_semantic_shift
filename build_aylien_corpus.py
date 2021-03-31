import json
from nltk import sent_tokenize, RegexpTokenizer
from collections import defaultdict
import argparse
import os

tokenizer = RegexpTokenizer('[\(]|[\w-]+|\$[\d\.]+|\S+')

def build_train_test(input_path, lm_output_train, lm_output_test):

    train = open(lm_output_train, 'w', encoding='utf8')
    test = open(lm_output_test, 'w', encoding='utf8')

    months = defaultdict(int)

    with open(input_path, 'r', encoding='utf8') as f:
        counter = 0
        for line in f:
            doc = json.loads(line)
            text = doc['body']
            text = " ".join(text.split())
            text = text.replace('“','').replace('”','').replace('"','')
            for sent in sent_tokenize(text):
                if counter % 10 == 0:
                    test.write(sent + '\n')
                else:
                    train.write(sent + '\n')
            time = doc["published_at"]
            month = time.split()[0].split('-')[1]
            months[month] += 1
            counter += 1
    print('LM train and test sets created.')

def build_data_sets(input_path, output_folder):

    january = open(os.path.join(output_folder,'aylien_january.txt'), 'w', encoding='utf8')
    february = open(os.path.join(output_folder, 'aylien_february.txt'), 'w', encoding='utf8')
    march = open(os.path.join(output_folder, 'aylien_march.txt'), 'w', encoding='utf8')
    april = open(os.path.join(output_folder, 'aylien_april.txt'), 'w', encoding='utf8')

    months = defaultdict(int)

    with open(input_path, 'r', encoding='utf8') as f:
        counter = 0
        for line in f:
            doc = json.loads(line)
            text = doc['body']
            text = " ".join(text.split())
            text = text.replace('“', '').replace('”', '').replace('"', '').strip()
            time = doc["published_at"]
            month = time.split()[0].split('-')[1]
            counter += 1
            text = {'content': text}
            line = json.dumps(text) + '\n'
            if str(month) == '01':
                january.write(line)
                months['jan'] += 1
            if str(month) == '02':
                february.write(line)
                months['feb'] += 1
            if str(month) == '03':
                march.write(line)
                months['mar'] += 1
            if str(month) == '04':
                april.write(line)
                months['apr'] += 1
        print("Num all papers: ", counter)
        print('months: ', months)
    january.close()
    february.close()
    march.close()
    april.close()
    print('Corpus slices created.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        help='Path to Aylien JSON file',
                        default='data/aylien/aylien-covid-news.jsonl')
    parser.add_argument('--output_folder', type=str, help='Path to output folder that will contain files for all temporal slices',
                        default='data/aylien')
    parser.add_argument('--lm_output_train', type=str,
                        help='Path to output .txt file used for language model training',
                        default='data/aylien/train.txt')
    parser.add_argument('--lm_output_test', type=str,
                        help='Path to output .txt file used for language model validation',
                        default='data/aylien/test.txt')
    args = parser.parse_args()

    build_train_test(args.input_path, args.lm_output_train, args.lm_output_test)
    build_data_sets(args.input_path, args.output_folder)
    print('D')


