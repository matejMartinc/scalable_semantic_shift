import pickle
import pandas as pd
import ot
import argparse
from scipy.spatial.distance import cdist

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from collections import Counter
from scipy.stats import entropy
from collections import defaultdict
import numpy as np
import os
import sys


def combine_clusters(labels, embeddings, threshold=10, remove=[]):
    cluster_embeds = defaultdict(list)
    for label, embed in zip(labels, embeddings):
        cluster_embeds[label].append(embed)
    min_num_examples = threshold
    legit_clusters = []
    for id, num_examples in Counter(labels).items():
        if num_examples >= threshold:
            legit_clusters.append(id)
        if id not in remove and num_examples < min_num_examples:
            min_num_examples = num_examples
            min_cluster_id = id

    if len(set(labels)) == 2:
        return labels

    min_dist = 1
    all_dist = []
    cluster_labels = ()
    embed_list = list(cluster_embeds.items())
    for i in range(len(embed_list)):
        for j in range(i+1,len(embed_list)):
            id, embed = embed_list[i]
            id2, embed2 = embed_list[j]
            if id in legit_clusters and id2 in legit_clusters:
                dist = compute_averaged_embedding_dist(embed, embed2)
                all_dist.append(dist)
                if dist < min_dist:
                    min_dist = dist
                    cluster_labels = (id, id2)

    std = np.std(all_dist)
    avg = np.mean(all_dist)
    limit = avg - 2 * std
    if min_dist < limit:
        for n, i in enumerate(labels):
            if i == cluster_labels[0]:
                labels[n] = cluster_labels[1]
        return combine_clusters(labels, embeddings, threshold, remove)

    if min_num_examples >= threshold:
        return labels


    min_dist = 1
    cluster_labels = ()
    for id, embed in cluster_embeds.items():
        if id != min_cluster_id:
            dist = compute_averaged_embedding_dist(embed, cluster_embeds[min_cluster_id])
            if dist < min_dist:
                min_dist = dist
                cluster_labels = (id, min_cluster_id)

    if cluster_labels[0] not in legit_clusters:
        for n, i in enumerate(labels):
            if i == cluster_labels[0]:
                labels[n] = cluster_labels[1]
    else:
        if min_dist < limit:
            for n, i in enumerate(labels):
                if i == cluster_labels[0]:
                    labels[n] = cluster_labels[1]
        else:
            remove.append(min_cluster_id)
    return combine_clusters(labels, embeddings, threshold, remove)


def compute_jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def cluster_word_embeddings_aff_prop(word_embeddings):
    clustering = AffinityPropagation().fit(word_embeddings)
    labels = clustering.labels_
    counts = Counter(labels)
    #print("Aff prop num of clusters:", len(counts))
    exemplars = clustering.cluster_centers_
    return labels, exemplars


def cluster_word_embeddings_k_means(word_embeddings, k, random_state):
    clustering = KMeans(n_clusters=k, random_state=random_state).fit(word_embeddings)
    labels = clustering.labels_
    exemplars = clustering.cluster_centers_
    return labels, exemplars


def compute_averaged_embedding_dist(t1_embeddings, t2_embeddings):
    t1_mean = np.mean(t1_embeddings, axis=0)
    t2_mean = np.mean(t2_embeddings, axis=0)
    dist = 1.0 - cosine_similarity([t1_mean], [t2_mean])[0][0]
    #print("Averaged embedding cosine dist:", dist)
    return dist


def compute_divergence_from_cluster_labels(embeds1, embeds2, labels1, labels2, threshold):
    labels_all = list(np.concatenate((labels1, labels2)))
    counts1 = Counter(labels1)
    counts2 = Counter(labels2)
    n_senses = list(set(labels_all))
    #print("Clusters:", len(n_senses))
    t1 = []
    t2 = []
    label_list = []
    for i in n_senses:
        if counts1[i] + counts2[i] > threshold:
            t1.append(counts1[i])
            t2.append(counts2[i])
            label_list.append(i)
    t1 = np.array(t1)
    t2 = np.array(t2)

    emb1_means = np.array([np.mean(embeds1[labels1 == clust], 0) for clust in label_list])
    emb2_means = np.array([np.mean(embeds2[labels2 == clust], 0) for clust in label_list])
    M = np.nan_to_num(np.array([cdist(emb1_means, emb2_means, metric='cosine')])[0], nan=1)
    t1_dist = t1 / t1.sum()
    t2_dist = t2 / t2.sum()
    wass = ot.emd2(t1_dist, t2_dist, M)
    jsd = compute_jsd(t1_dist, t2_dist)
    return jsd, wass


def detect_meaning_gain_and_loss(labels1, labels2, threshold):
    labels1 = list(labels1)
    labels2 = list(labels2)
    all_count = Counter(labels1 + labels2)
    first_count = Counter(labels1)
    second_count = Counter(labels2)
    gained_meaning = False
    lost_meaning = False
    all = 0
    meaning_gain_loss = 0

    for label, c in all_count.items():
        all += c
        if c >= threshold:
            if label not in first_count or first_count[label] <= 2:
                gained_meaning=True
                meaning_gain_loss += c
            if label not in second_count or second_count[label] <= 2:
                lost_meaning=True
                meaning_gain_loss += c
    return str(gained_meaning) + '/' + str(lost_meaning), meaning_gain_loss/all


def compute_divergence_across_many_periods(embeddings, labels, splits, corpus_slices, threshold, method):
    all_clusters = []
    all_embeddings = []
    clusters_dict = {}
    for split_num, split in enumerate(splits):
        if split_num > 0:
            clusters = labels[splits[split_num-1]:split]
            clusters_dict[corpus_slices[split_num - 1]] = clusters
            all_clusters.append(clusters)
            ts_embeds = embeddings[splits[split_num - 1]:split]
            all_embeddings.append(ts_embeds)
    all_measures = []
    all_meanings = []
    for i in range(len(all_clusters)):
        if i < len(all_clusters) -1:
            try:
                jsd, wass = compute_divergence_from_cluster_labels(all_embeddings[i],all_embeddings[i+1], all_clusters[i],all_clusters[i+1], threshold)
            except:
                jsd, wass = 0, 0
            meaning, meaning_score = detect_meaning_gain_and_loss(all_clusters[i],all_clusters[i+1], threshold)
            all_meanings.append(meaning)
            if method == 'WS':
                measure = wass
            else:
                measure = jsd
            all_measures.append(measure)
    try:
        entire_jsd, entire_wass = compute_divergence_from_cluster_labels(all_embeddings[0],all_embeddings[-1], all_clusters[0],all_clusters[-1], threshold)
    except:
        entire_jsd, entire_wass = 0, 0
    meaning, meaning_score = detect_meaning_gain_and_loss(all_clusters[0],all_clusters[-1], threshold)
    all_meanings.append(meaning)


    avg_measure = sum(all_measures)/len(all_measures)
    try:
        measure = entire_wass
    except:
        measure = 0
    all_measures.extend([measure, avg_measure])
    all_measures = [float("{:.6f}".format(score)) for score in all_measures]
    return all_measures, all_meanings, clusters_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure semantic shift')
    parser.add_argument("--method", default='WD', const='all', nargs='?',
                        help="A method for calculating distance", choices=['WD', 'JSD'])
    parser.add_argument("--corpus_slices",
                        default='1960;1990',
                        type=str,
                        help="Time slices names separated by ';'.")
    parser.add_argument("--get_additional_info", action="store_true", help='Whether the cluster labels and sentences, required for interpretation, are saved or not.')
    parser.add_argument('--results_dir_path', type=str, default='results_coha', help='Path to the folder to save the results.')
    parser.add_argument('--embeddings_path', type=str, default='embeddings/coha_fine_tuned_scalable.pickle', help='Path to the embeddings pickle file.')
    parser.add_argument('--define_words_to_interpret', type=str, default='', help='Define a set of words separated by ";" for interpretation if you do not wish to save data for all words.')
    parser.add_argument('--random_state', type=int, default=123, help='Choose a random state for reproducibility of clustering.')
    parser.add_argument('--cluster_size_threshold', type=int, default=10, help='Clusters smaller than a threshold will be merged or deleted.')
    args = parser.parse_args()
    random_state = args.random_state
    threshold = args.cluster_size_threshold
    get_additional_info = args.get_additional_info
    embeddings_file = args.embeddings_path
    corpus_slices = args.corpus_slices


    methods = ['WD', 'JSD']
    if args.method not in methods:
        print("Method not valid, valid choices are: ", ", ".join(methods))
        sys.exit()

    print("Loading ", embeddings_file)
    try:
        bert_embeddings, count2sents = pickle.load(open(embeddings_file, 'rb'))
    except:
        bert_embeddings = pickle.load(open(embeddings_file, 'rb'))
        count2sents = None

    if len(args.define_words_to_interpret) > 0:
        target_words = args.define_words_to_interpret.split(';')
    else:
        target_words = list(bert_embeddings.keys())

    if get_additional_info and len(target_words) > 100:
        print('Define a list of words to interpret with less than 100 words or set "get_additional_info" flag to False')
        sys.exit()

    measure_vec = []
    cosine_dist_vec = []
    sentence_dict = {}
    aff_prop_labels_dict = {}
    aff_prop_centroids_dict = {}
    kmeans_5_labels_dict = {}
    kmeans_5_centroids_dict = {}
    kmeans_7_labels_dict = {}
    kmeans_7_centroids_dict = {}

    aff_prop_pref = -430
    print("Clustering BERT embeddings")
    print("Len target words: ", len(target_words))

    results = []

    print("Words in embeds: ", bert_embeddings.keys())

    for i, word in enumerate(target_words):
        print("\n=======", i + 1, "- word:", word.upper(), "=======")

        if word not in bert_embeddings:
            continue
        emb = bert_embeddings[word]
        if i == 0:
            print("Time periods in embeds: ", emb.keys())

        all_embeddings = []
        all_sentences = {}
        splits = [0]
        all_slices_present = True
        all_freqs = []

        cs_counts = []

        for cs in corpus_slices:
            cs_embeddings = []
            cs_sentences = []

            count_all = 0
            text_seen = set()

            if cs not in emb:
                all_slices_present = False
                print('Word missing in slice: ', cs)
                continue

            counts = [x[1] for x in emb[cs]]
            cs_counts.append(sum(counts))
            all_freqs.append(sum(counts))
            cs_text = cs + '_text'
            print("Slice: ", cs)
            print("Num embeds: ", len(emb[cs]))
            num_sent_codes = 0

            for idx in range(len(emb[cs])):

                #get summed embedding and its count, devide embedding by count
                try:
                    e, count_emb = emb[cs][idx]
                    e = e/count_emb
                except:
                    e = emb[cs][idx]

                sents = set()

                #print("Num sentences: ", len(sent_codes))
                if count2sents is not None:
                    sent_codes = emb[cs_text][idx]
                    num_sent_codes += len(sent_codes)
                    for sent in sent_codes:
                        if sent in count2sents[cs]:
                            text = count2sents[cs][sent]

                        sents.add(text)
                        #print(text)

                cs_embeddings.append(e)
                cs_sentences.append(" ".join(list(sents)))

            all_embeddings.append(np.array(cs_embeddings))
            all_sentences[cs] = cs_sentences
            splits.append(splits[-1] + len(cs_embeddings))


        print("Num all sents: ", num_sent_codes)
        print("Num words in corpus slice: ", cs_counts)
        embeddings_concat = np.concatenate(all_embeddings, axis=0)

        #we can not use kmeans7 if there are less than 7 examples
        if embeddings_concat.shape[0] < 7 or not all_slices_present:
            continue
        else:
            aff_prop_labels, aff_prop_centroids = cluster_word_embeddings_aff_prop(embeddings_concat)
            aff_prop_labels = combine_clusters(aff_prop_labels, embeddings_concat, threshold=threshold, remove=[])
            all_aff_prop_measures, all_meanings, clustered_aff_prop_labels = compute_divergence_across_many_periods(embeddings_concat, aff_prop_labels, splits, corpus_slices, threshold, args.method)
            kmeans_5_labels, kmeans_5_centroids = cluster_word_embeddings_k_means(embeddings_concat, 5, random_state)
            kmeans_5_labels = combine_clusters(kmeans_5_labels, embeddings_concat, threshold=threshold, remove=[])
            all_kmeans5_measures, all_meanings, clustered_kmeans_5_labels = compute_divergence_across_many_periods(embeddings_concat, kmeans_5_labels, splits, corpus_slices, threshold, args.method)
            kmeans_7_labels, kmeans_7_centroids = cluster_word_embeddings_k_means(embeddings_concat, 7, random_state)
            kmeans_7_labels = combine_clusters(kmeans_7_labels, embeddings_concat, threshold=threshold, remove=[])
            all_kmeans7_measures, all_meanings, clustered_kmeans_7_labels = compute_divergence_across_many_periods(embeddings_concat, kmeans_7_labels, splits, corpus_slices, threshold, args.method)
            all_freqs = all_freqs + [sum(all_freqs)] + [sum(all_freqs)/len(all_freqs)]
            word_results = [word] +  all_aff_prop_measures + all_kmeans5_measures + all_kmeans7_measures + all_freqs + all_meanings
            print("Results:", word_results)
        results.append(word_results)

        #add results to dataframe for saving
        if get_additional_info:
            sentence_dict[word] = all_sentences
            aff_prop_labels_dict[word] = clustered_aff_prop_labels
            aff_prop_centroids_dict[word] = aff_prop_centroids

            kmeans_5_labels_dict[word] = clustered_kmeans_5_labels
            kmeans_5_centroids_dict[word] = kmeans_5_centroids

            kmeans_7_labels_dict[word] = clustered_kmeans_7_labels
            kmeans_7_centroids_dict[word] = kmeans_7_centroids  # add results to dataframe for saving

    columns = ['word']
    methods = ['AP', 'K5', 'K7', 'FREQ', 'MEANING GAIN/LOSS']
    for method in methods:
        for num_slice, cs in enumerate(corpus_slices):
            if method == 'FREQ':
                columns.append(method + ' ' + cs)
            else:
                if num_slice < len(corpus_slices) - 1:
                    columns.append(method + ' ' + cs + '-' + corpus_slices[num_slice + 1])
        columns.append(method + ' All')
        if method != 'MEANING GAIN/LOSS':
            columns.append(method + ' Avg')


    if not os.path.exists(args.results_dir_path):
        os.makedirs(args.results_dir_path)

    csv_file = args.results_dir_path + "word_ranking_results_" + args.method + ".csv"

    # save results to CSV
    results_df = pd.DataFrame(results, columns=columns)
    results_df = results_df.sort_values(by=['K5 Avg'], ascending=False)
    results_df.to_csv(csv_file, sep=';', encoding='utf-8', index=False)

    print("Done! Saved results in", csv_file, "!")

    if get_additional_info:

        # save cluster labels and sentences to pickle
        dicts = [(aff_prop_centroids_dict, 'aff_prop_centroids'), (aff_prop_labels_dict, 'aff_prop_labels'),
                 (kmeans_5_centroids_dict, 'kmeans_5_centroids'), (kmeans_5_labels_dict, 'kmeans_5_labels'),
                 (kmeans_7_centroids_dict, 'kmeans_7_centroids'), (kmeans_7_labels_dict, 'kmeans_7_labels'),
                 (sentence_dict, "sents")]

        for data, name in dicts:
            data_file = os.path.join(args.results_dir_path, name + ".pkl")
            pf = open(data_file, 'wb')
            pickle.dump(data, pf)
            pf.close()

