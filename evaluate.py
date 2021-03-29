from scipy.stats.stats import spearmanr
import sys
import pandas as pd
import argparse


def get_durel_shifts(input_path):
    shifts_dict = {}
    df_shifts = pd.read_csv(input_path, sep='\t', encoding='utf8')
    for idx, row in df_shifts.iterrows():
        shifts_dict[row['Lexeme']] = float(row['LSC'])
    return shifts_dict


def get_coha_shifts(input_path):
    shifts_dict = {}
    df_shifts = pd.read_csv(input_path, sep=',', encoding='utf8')
    for idx, row in df_shifts.iterrows():
        shifts_dict[row['word']] = row['mean']
    return shifts_dict


def get_semeval_shifts(input_path):
    shifts_dict = {}
    with open(input_path, 'r', encoding='utf8') as f:
        for line in f:
            word, score = line.split('\t')
            word = word.strip()
            score = float(score.strip())
            shifts_dict[word] = score


def evaluate(result_path, gold_standard_shifts_path, task, slices):
    df = pd.read_csv(result_path, encoding='utf8', sep=';')
    df = df.sort_values('word')

    aff = df[f'AP {slices}'].tolist()
    kmeans5 = df[f'K5 {slices}'].tolist()
    kmeans7 = df[f'K7 {slices}'].tolist()

    if task == 'durel':
        gs = get_durel_shifts(gold_standard_shifts_path).items()
    elif task == 'coha':
        gs = get_coha_shifts(gold_standard_shifts_path).items()
    elif task == 'semeval':
        gs = get_semeval_shifts(gold_standard_shifts_path).items()

    gs = sorted(gs, key=lambda x: x[0])
    gs_shifts = [x[1] for x in gs]

    return [spearmanr(aff, gs_shifts)[0], spearmanr(kmeans5, gs_shifts)[0], spearmanr(kmeans7, gs_shifts)[0]]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='coha', const='all', nargs='?',
                        help="Choose a task", choices=['coha', 'semeval', 'durel'])
    parser.add_argument("--gold_standard_path", default='data/coha/Gulordava_word_meaning_change_evaluation_dataset.csv', type=str,
                        help="Path to gold standard file")
    parser.add_argument('--results_dir_path', type=str, default='results_coha/word_ranking_results_WS.csv', help='Path to the folder to save the results.')
    parser.add_argument("--corpus_slices_names",
                        default='1960;1990',
                        type=str,
                        help="Time slices names separated by ';'.")
    args = parser.parse_args()
    tasks = ['coha', 'semeval', 'durel']
    if args.task not in tasks:
        print("Task not valid, valid choices are: ", ", ".join(tasks))
        sys.exit()


    aff_prop, kmeans5, kmeans7 = evaluate(args.results_dir_path, args.gold_standard_path, args.task, "-".join(args.corpus_slices_names.split(';')))
    print(f'Results for {args.task}:')
    print('K-means 5:', kmeans5)
    print('K-means 7:', kmeans7)
    print('Aff-prop:', aff_prop)





























