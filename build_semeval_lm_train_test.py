import random
import argparse
import os
import sys

def filterLine(line, lang, targets):
    if lang=='latin':
        line = ''.join([i for i in line if not (i.isdigit() or i=='#')])
    elif lang=='english':
        wrong_pos = False
        correct_pos = False
        for target in targets:
            line_l = line.split()
            if target in line:
                line = line.replace(target, target[:-3])
                correct_pos = True
            if target[:-3] in line_l:
                wrong_pos = True
        if correct_pos:
            return line
        elif wrong_pos:
            return None
    return line


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_paths", default='data/english/english_1.txt;data/english/english_2.txt', type=str,
                        help="Paths to all corpus time slices separated by ';'.")
    parser.add_argument("--target_path", default='data/english/targets.txt', type=str,
                        help="Path to gold standard word target files")
    parser.add_argument("--language", const='english', nargs='?',
                        help="Choose a language", choices=['english', 'latin', 'swedish', 'german'])
    parser.add_argument("--lm_train_test_folder", default='data/english',
                        help="Path to folder that contains output language model train and test sets")
    args = parser.parse_args()

    lang = args.language
    languages = ['english', 'latin', 'swedish', 'german']
    if lang not in languages:
        print("Language not valid, valid choices are: ", ", ".join(languages))
        sys.exit()

    target_path = args.target_path
    corpora = args.corpus_paths.split(';')
    output_folder = args.lm_train_test_folder
    data = []
    if lang == 'english':
        targets = []
        with open(target_path, 'r', encoding='utf8') as f:
            for line in f:
                target = line.strip()
                if len(target) > 0 :
                    targets.append(target)
    else:
        targets = None

    for corpus in corpora:
        with open(corpus, 'r', encoding='utf8') as f:
            for line in f:
                line = filterLine(line, lang, targets)
                if line is not None:
                    data.append(line)

    random.shuffle(data)
    valid_index = int(0.9 * len(data))

    output_train = open(os.path.join(output_folder,'train.txt'), 'w', encoding='utf8')
    output_test = open(os.path.join(output_folder, 'test.txt'), 'w', encoding='utf8')

    for idx, sent in enumerate(data):
        if idx < valid_index:
            output_train.write(sent)
        else:
            output_test.write(sent)

    output_train.close()
    output_test.close()

