import random
import argparse

def build_train_test(input_files, lm_output_train, lm_output_test):
    data = []
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf8') as f:
            for line in f:
                if line is not None:
                    data.append(line)

    random.shuffle(data)
    valid_index = int(0.9 * len(data))


    output_train = open(lm_output_train, 'w', encoding='utf8')
    output_test = open(lm_output_test, 'w', encoding='utf8')

    for idx, sent in enumerate(data):
        if idx < valid_index:
            output_train.write(sent)
        else:
            output_test.write(sent)

    output_train.close()
    output_test.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str,
                        help='Path to DUREL files containing text for each temporal slice separated by ";".',
                        default='data/durel/dta_18.txt;data/durel/dta_19.txt')
    parser.add_argument('--lm_output_train', type=str,
                        help='Path to output .txt file used for language model training',
                        default='data/durel/train.txt')
    parser.add_argument('--lm_output_test', type=str,
                        help='Path to output .txt file used for language model validation',
                        default='data/durel/test.txt')
    args = parser.parse_args()

    input_files = args.input_files.split(';')

    build_train_test(input_files, args.lm_output_train, args.lm_output_test)
    print('Done, language model train and test sets written to:', args.lm_output_train, args.lm_output_test)

