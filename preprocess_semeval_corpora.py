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
                        help="Path to target files")
    parser.add_argument("--language", default='english', const='all', nargs='?',
                        help="Choose a language", choices=['english', 'latin', 'swedish', 'german'])
    parser.add_argument("--output_folder", default='data/english',
                        help="Path to folder that contains output preprocessed files, one per slice")
    args = parser.parse_args()
    lang = args.language
    languages = ['english', 'latin', 'swedish', 'german']
    if lang not in languages:
        print("Language not valid, valid choices are: ", ", ".join(languages))
        sys.exit()
    target_path = args.target_path
    corpora = args.corpus_paths.split(';')
    output_folder = args.output_folder
    data = []
    outputs = []

    for i, corpus in enumerate(corpora):
        print(lang, output_folder)
        output = open(os.path.join(output_folder, lang + '_preprocessed_' + str(i + 1) + '.txt'), 'w', encoding='utf8')
        outputs.append(output)

    if lang == 'english':
        targets = []
        with open(target_path, 'r', encoding='utf8') as f:
            for line in f:
                target = line.strip()
                if len(target) > 0 :
                    targets.append(target)
    else:
        targets = None

    for i, corpus in enumerate(corpora):
        with open(corpus, 'r', encoding='utf8') as f:
            for line in f:
                line = filterLine(line, lang, targets)
                if line is not None:
                    outputs[i].write(line)

    for output in outputs:
        output.close()


