from scipy.stats.stats import spearmanr
import pandas as pd

def get_shifts(input_path):
    shifts_dict = {}
    df_shifts = pd.read_csv(input_path, sep=',', encoding='utf8')
    for idx, row in df_shifts.iterrows():
        shifts_dict[row['word']] = row['mean']
    return shifts_dict


folder= 'coha_results'
folder_results = 'test_data_truth/task2'

path = 'coha_results/results_english_fine_tuned_averaged.csv'
df = pd.read_csv(path, encoding='utf8', sep='\t')
df = df.sort_values('word')
print(df)
aff = df['aff_prop'].tolist()
kmeans5 = df['kmeans_5'].tolist()
kmeans7 = df['kmeans_7'].tolist()
avg = df['averaging'].tolist()




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

print('Averaging')
spearman = spearmanr(avg, gs_shifts)[0]
print('Spearman: ', spearman)

'''
baseline pretrained
Aff prop
Spearman:  0.4132669949204701
kmeans 5
Spearman:  0.423759127654016
kmeans 7
Spearman:  0.43089514496595216
Averaging
Spearman:  0.4013052800776652
baseline fine-tuned
Aff prop
Spearman:  0.38152402136047836
kmeans 5
Spearman:  0.45866682919813273
kmeans 7
Spearman:  0.455994240338969
Averaging
Spearman:  0.4136771108579377
baseline fine-tuned 128
Aff prop
Spearman:  0.4351535154499907
kmeans 5
Spearman:  0.4742854111500236
kmeans 7
Spearman:  0.46448364024454813
Averaging
Spearman:  0.4179218108107273

baseline fine-tuned sentence only
Aff prop
Spearman:  0.42157184265418884
kmeans 5
Spearman:  0.4751398193530811
kmeans 7
Spearman:  0.489049584898857
Averaging
Spearman:  0.3941214159063579

Aff prop
Spearman:  0.4315103188721536
kmeans 5
Spearman:  0.47002704066598516
kmeans 7
Spearman:  0.49314390900790844
Averaging
Spearman:  0.416342864451477
'''


'''
20
Aff prop
Spearman:  0.28021854954035813
kmeans 5
Spearman:  0.28463413113375924
kmeans 7
Spearman:  0.328653241755281
Averaging
Spearman:  0.2972315256796388
'''

'''
200
Aff prop
Spearman:  0.34744338695692123
kmeans 5
Spearman:  0.416431722904595
kmeans 7
Spearman:  0.36542013554925085
Averaging
Spearman:  0.34285692372290866
'''

'''
100 fine-tuned
Aff prop
Spearman:  0.28295265579014206
kmeans 5
Spearman:  0.2832875838057406
kmeans 7
Spearman:  0.23807230169993837
Averaging
Spearman:  0.29337643586744344

'''

'''
Aff prop 200 treshold 0.9
Spearman:  0.4448390868398504
kmeans 5
Spearman:  0.4301501010128861
kmeans 7
Spearman:  0.3892410362504935
Averaging
Spearman:  0.32551585483365386

Aff prop 200 treshold 0.9 sentence only
Spearman:  0.39161287342218115
kmeans 5
Spearman:  0.44003389510585506
kmeans 7
Spearman:  0.3823032416416668
Averaging
Spearman:  0.361722256846418
'''

'''
kmeans stuff, divide and conquer, pretrained
Aff prop
Spearman:  0.2949143706329469
kmeans 5
Spearman:  0.3238275442244123
kmeans 7
Spearman:  0.32195468144331024
Averaging
Spearman:  0.3313053248175714
fine tuned
Spearman:  0.22168816999810798
kmeans 5
Spearman:  0.2442718876213235
kmeans 7
Spearman:  0.30354731111663974
Averaging
Spearman:  0.20788093343669897
'''













