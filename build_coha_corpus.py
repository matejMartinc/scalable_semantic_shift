import json
import nltk
import os
import re
from collections import defaultdict

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

count_words = {}


def remove_mentions(text, replace_token):
    return re.sub(r'(?:@[\w_]+)', replace_token, text)


def build_train_test():
    output_train = open('./train.txt', 'w', encoding='utf')
    output_test = open('./test.txt', 'w', encoding='utf')

    input_folders = ['./COHA_1960', './COHA_1990']
    all_sents = 0
    for input_folder in input_folders:
        num_docs = len(os.listdir(input_folder))
        limit = int(0.9 * num_docs)
        counter = 0
        for file in os.listdir(input_folder):

            path = os.path.join(input_folder, file)

            with open(path, 'r', encoding='utf8') as f:
                text = f.read()
                text = remove_mentions(text, '')
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
    print(all_sents)

    output_train.close()
    output_test.close()




#build_train_test()

def build_data_sets():
    years = [1960, 1990]
    input_folders = ['data/coha/COHA_1960', 'data/coha/COHA_1990']

    datasets = ['data/coha/coha_1960.txt',
                'data/coha/coha_1990.txt']

    for ds, year in zip(datasets,years):

        output = open(ds, 'w', encoding='utf')
        for input_folder in input_folders:
            if str(year) in input_folder:
                for file in os.listdir(input_folder):

                    path = os.path.join(input_folder, file)

                    with open(path, 'r', encoding='utf8') as f:
                        text = f.read()
                        #text = remove_mentions(f.read(), '')
                        #text = text.replace('<p>', '')
                        #text = text.replace('<P>', '')
                        #text = text.replace('@', '')
                        #text = text.replace('"', '')
                        text = " ". join(text.split())
                        text = {'content': text, 'title': ''}
                        line = json.dumps(text) + '\n'
                        output.write(line)
        output.close()

def get_stats():

    input_folders = ['./COHA_1960', './COHA_1990']

    for input_folder in input_folders:
        dd = defaultdict(int)
        for file in os.listdir(input_folder):
            path = os.path.join(input_folder, file)

            with open(path, 'r', encoding='utf8') as f:
                text = remove_mentions(f.read(), '')
                text = text.replace('<p>', '')
                text = text.replace('<P>', '')
                text = text.replace('@', '')
                text = text.replace('"', '')
                for word in text.split():
                    if word in ['assay', 'sulphate', 'mediaeval', 'extracellular']:
                        dd[word] += 1
        print(dd)


#get_stats()
#build_train_test()
build_data_sets()





