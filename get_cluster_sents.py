import pickle
from collections import defaultdict, Counter
from pprint import pprint
import sys


if __name__ == '__main__':

    target = 'economy'
    labels = "aylien_results/aff_prop_labels_english_fine_tuned_averaged.pkl"
    sentences = "aylien_results/sents_english_fine_tuned_averaged.pkl"
    corpus_slices = ['fox', 'cnn']

    labels = pickle.load(open(labels, 'rb'))
    sentences = pickle.load(open(sentences, 'rb'))

    cluster_to_sentence = defaultdict(lambda: defaultdict(list))
    for cs in corpus_slices:
        for label, sent in zip(labels[target][cs], sentences[target][cs]):
            cluster_to_sentence[label][cs].append(sent)

    counts = {cs: Counter(labels[target][cs]) for cs in corpus_slices}
    common_counts = []
    all_labels = []
    for slice, c in counts.items():
        slice_labels = [x[0] for x in c.items()]
        all_labels.extend(slice_labels)
    all_labels = set(all_labels)

    all_counts = []


    for l in all_labels:
        all_count = 0
        for slice in corpus_slices:
            count = counts[slice][l]
            all_count += count
        all_counts.append((l, all_count))



    sorted_counts = sorted(all_counts, key=lambda x: x[1], reverse=True)

    for label, count in sorted_counts:
        print("\n================================\n")

        print("Cluster label: ", label)
        print("Cluster size: ", count)

        for cs in corpus_slices:
            print("Corpus slice: ", cs)
            print("Num appearances: ", counts[cs].get(label, 0))
            print()
            for s in cluster_to_sentence[label][cs]:
                print(s.replace("[CLS]", "").replace("[SEP]", "").strip())



