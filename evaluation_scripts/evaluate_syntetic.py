from scipy.stats.stats import spearmanr
import pandas as pd

def get_shifts(input_path):
    shifts_dict = {}
    df_shifts = pd.read_csv(input_path, sep=',', encoding='utf8')
    for idx, row in df_shifts.iterrows():
        shifts_dict[row['word']] = row['mean']
    return shifts_dict


path_preds = 'syntetic_results/results_english_fine_tuned_averaged.csv'
path_gs = 'data/syntetic_data/syntetic_gs.csv'
df_preds = pd.read_csv(path_preds, encoding='utf8', sep=';')
df_preds = df_preds.sort_values('word')
df_gs = pd.read_csv(path_gs, encoding='utf8', sep=';')
df_gs = df_gs.sort_values('word')


algorithms = ['AP', 'K5', 'K7']

averages_k5 = []
averages_k7 = []
averages_ap = []

for gs_column in df_gs.columns:
    if gs_column != 'word':
        ts = gs_column.split()[-1]
        for pred_column in df_preds.columns:
            #print(pred_column)
            if ts in pred_column or (ts=='first-last' and 'All' in pred_column):
                if pred_column != 'word':
                    #print(pred_column)
                    alg = pred_column.split()[1]
                    if alg in algorithms:
                        data_gs = df_gs[gs_column].tolist()
                        data_preds = df_preds[pred_column].tolist()
                        spearman = spearmanr(data_preds, data_gs)[0]
                        if ts !='first-last':
                            if alg == 'AP':
                                averages_ap.append(spearman)
                            if alg == 'K5':
                                averages_k5.append(spearman)
                            if alg == 'K7':
                                averages_k7.append(spearman)

                        print(alg, ts, spearman)
        print('-------------------------------------------------------')
        print()

print('Averages:')
print("AP: ", sum(averages_ap)/len(averages_ap))
print("K5: ", sum(averages_k5)/len(averages_k5))
print("K7: ", sum(averages_k7)/len(averages_k7))




