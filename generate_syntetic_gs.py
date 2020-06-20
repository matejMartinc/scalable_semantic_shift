import numpy as np
import torch
import sys
import pickle
import dill
from measure_semantic_shift import compute_jsd
import pandas as pd


def get_syntetic_targets():
    data = []
    path = 'data/syntetic_data/english_gold_standard_dict.pkl'
    gs = dill.load(open(path, 'rb'))
    #targets = gs.keys()
    time_slices = [str(i) for i in range(10)]
    columns = ['word']
    word_counter = 0

    for target, senses in gs.items():
        word_counter += 1
        row = [target.lower()]

        for idx in range(len(time_slices) - 1):
            ts1 = list(senses[time_slices[idx]].items())
            ts1 = sorted(ts1, key=lambda x: x[0])
            ts1_dist = [x[1] for x in ts1]
            ts2 = list(senses[time_slices[idx + 1]].items())
            ts2 = sorted(ts2, key=lambda x: x[0])
            ts2_dist = [x[1] for x in ts2]
            jsd = compute_jsd(ts1_dist, ts2_dist)
            if word_counter == 1:
                columns.append('JSD ' + '0' + time_slices[idx] + '-0' + time_slices[idx + 1])
            row.append(jsd)

        ts_first = list(senses[time_slices[0]].items())
        ts_first = sorted(ts_first, key=lambda x: x[0])
        ts_first_dist = [x[1] for x in ts_first]
        ts_last = list(senses[time_slices[-1]].items())
        ts_last = sorted(ts_last, key=lambda x: x[0])
        ts_last_dist = [x[1] for x in ts_last]
        jsd = compute_jsd(ts_first_dist, ts_last_dist)
        if word_counter == 1:
            columns.append('JSD first-last')
        row.append(jsd)
        data.append(row)


    df = pd.DataFrame(data, columns=columns)
    df = df.sort_values(by=['JSD first-last'], ascending=False)
    df.to_csv('syntetic_gs.csv', encoding='utf8', sep=';', index=False)


get_syntetic_targets()
