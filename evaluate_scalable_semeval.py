from scipy.stats.stats import spearmanr
import os
import pandas as pd

folder= 'semeval_scalable_results'
folder_results = 'test_data_truth/task2'

paths = []

for root, dirs, files in os.walk(folder, topdown=False):
   for name in files:
       p = os.path.join(root, name)
       if p.endswith('csv') and p not in paths:
           paths.append(p)
   for name in dirs:
       p = os.path.join(root, name)
       if p.endswith('csv')  and p not in paths:
           paths.append(p)

dict = {}
for path in paths:
    df = pd.read_csv(path, encoding='utf8', sep=';')
    df = df.sort_values('word')
    aff = df['JSD AP 1-2'].tolist()
    kmeans5 = df['JSD K5 1-2'].tolist()
    kmeans7 = df['JSD K7 1-2'].tolist()
    path = path.split('/')[-1][:-4]

    dict[path] = [aff, kmeans5, kmeans7]


result_dict={}
for p in os.listdir(folder_results):
    l = []
    with open(os.path.join(folder_results,p), 'r', encoding='utf8') as f:
        for line in f:
            word, score = line.split('\t')
            word = word.strip()
            score = float(score.strip())
            l.append((word, score))
    l = sorted(l, key=lambda x: x[0])
    result_dict[p] = [x[1] for x in l]
    #print('gold standard: ', p)
    k = sorted(l, key=lambda x: x[1], reverse=True)
    #for w, s in k:
        #print(w,s)

    #print('-------------------------------------------------------')
    #print()



spearman_l = []
spearman_d = {}

algs = ['affprop', 'kmeans5', 'kmeans7']

for i in range(3):
    for path, values in dict.items():
        for p, v in result_dict.items():
            same_lang = False

            if 'english' in p and 'english' in path:
                same_lang = True
            if 'latin' in p and 'latin' in path:
                same_lang = True
            if 'german' in p and 'german' in path:
                same_lang = True
            if 'swedish' in p and 'swedish' in path:
                same_lang = True
            if same_lang:
                print(p, path)
                spearman = spearmanr(values[i], v)[0]
                print(spearman)
                spearman_l.append((path + '_' + algs[i], spearman))
                spearman_d[path + '_' + algs[i]] = spearman

spearman_l = sorted(spearman_l, reverse=True, key=lambda x: x[1])

averages = []



for s in spearman_l:
    print(s[0], ':', s[1])
    name = s[0]
    if 'english' in name:
        value = s[1]
        name = name.replace('english', 'german')
        g_value = spearman_d[name]
        name = name.replace('german', 'swedish')
        s_value = spearman_d[name]
        name = name.replace('swedish', 'latin')
        l_value = spearman_d[name]
        avg = sum([value, g_value, s_value, l_value])/4
        name = name.replace('latin_', '')
        averages.append((name, avg))


averages = sorted(averages, reverse=True, key=lambda x: x[1])

print('Averages')

for i, avg in enumerate(averages):
    print(avg[0] + ':', avg[1])



'''
results_german_fine_tuned_averaged_kmeans5 : 0.5270386948241076
results_german_fine_tuned_averaged_kmeans7 : 0.5124591768417727
results_german_fine_tuned_averaged_affprop : 0.49793406004593904
results_english_fine_tuned_averaged_affprop : 0.3705021103109672
results_latin_fine_tuned_averaged_affprop : 0.346451522497557
results_latin_fine_tuned_averaged_kmeans5 : 0.3422299361145648
results_latin_fine_tuned_averaged_kmeans7 : 0.34054130156136797
results_english_fine_tuned_averaged_kmeans5 : 0.312763061440106
results_english_fine_tuned_averaged_kmeans7 : 0.29201494326269184
results_swedish_fine_tuned_averaged_kmeans5 : 0.10036489073193285
results_swedish_fine_tuned_averaged_kmeans7 : 0.09611557076142764
results_swedish_fine_tuned_averaged_affprop : 0.011533868491371314
Averages
results_fine_tuned_averaged_kmeans5: 0.32059914577767784
results_fine_tuned_averaged_kmeans7: 0.310282748106815
results_fine_tuned_averaged_affprop: 0.3066053903364586
'''












