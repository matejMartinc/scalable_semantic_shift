from scipy.stats.stats import spearmanr
import pandas as pd

def get_shifts(input_path):
    shifts_dict = {}
    df_shifts = pd.read_csv(input_path, sep=',', encoding='utf8')
    for idx, row in df_shifts.iterrows():
        shifts_dict[row['word']] = row['mean']
    return shifts_dict



path = 'coha_results/results_english_fine_tuned_averaged.csv'
df = pd.read_csv(path, encoding='utf8', sep=';')
df = df.sort_values('word')
print(df)
aff = df['JSD AP 1960-1990'].tolist()
kmeans5 = df['JSD K5 1960-1990'].tolist()
kmeans7 = df['JSD K7 1960-1990'].tolist()

gs = get_shifts('data/coha/Gulordava_word_meaning_change_evaluation_dataset.csv').items()
gs = sorted(gs, key=lambda x: x[0])
gs_shifts = [x[1] for x in gs]

for w, s in gs:
    print(w,s)
print('-------------------------------------------------------')
print()




print('Aff prop')
spearman = spearmanr(aff, gs_shifts)[0]
print('Spearman: ', spearman)

print('kmeans 5')
spearman = spearmanr(kmeans5, gs_shifts)[0]
print('Spearman: ', spearman)

print('kmeans 7')
spearman = spearmanr(kmeans7, gs_shifts)[0]
print('Spearman: ', spearman)



'''scikit 0.21.3 200 0.99
Aff prop
Spearman:  0.3936839589063925
kmeans 5
Spearman:  0.4296989734816717
kmeans 7
Spearman:  0.4646681924164085
'''















