import pickle
import pandas as pd
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from collections import Counter
from scipy.stats import entropy
import numpy as np
import os
import sys
import ot


def compute_jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2

def filter_english(text, word):
    if word in text and word[:-3] not in text.split():
        return False
    else:
        print(text, word)
        return True

def cluster_word_embeddings_aff_prop(word_embeddings):
    clustering = AffinityPropagation().fit(word_embeddings)
    labels = clustering.labels_
    counts = Counter(labels)
    print("Aff prop num of clusters:", len(counts))
    exemplars = clustering.cluster_centers_
    return labels, exemplars


def cluster_word_embeddings_dbscan(word_embeddings):
    clustering = DBSCAN().fit(word_embeddings)
    labels = clustering.labels_
    counts = Counter(labels)
    print("DBSCAN num of clusters:", len(counts))
    return labels


def cluster_word_embeddings_k_means(word_embeddings, k, random_state):
    clustering = KMeans(n_clusters=k, random_state=random_state).fit(word_embeddings)
    labels = clustering.labels_
    exemplars = clustering.cluster_centers_
    return labels, exemplars


def compute_mean_dist(t1_embeddings, t2_embeddings):
    t1_len = t1_embeddings.shape[0]
    t2_len = t2_embeddings.shape[0]
    mean_overall = []
    for t1_i in range(t1_len):
        mean_i = []
        for t2_i in range(t2_len):
            dist = 1.0 - (cosine_similarity([t1_embeddings[t1_i]], [t2_embeddings[t2_i]])[0][0])
            mean_i.append(dist)
        mean_i = np.mean(mean_i)
        # print("Mean for instance:", mean_i)
        mean_overall.append(mean_i)
    mean_overall = np.mean(mean_overall)
    print("Mean cosine dist:", mean_overall)


def compute_averaged_embedding_dist(t1_embeddings, t2_embeddings):
    t1_mean = np.mean(t1_embeddings, axis=0)
    t2_mean = np.mean(t2_embeddings, axis=0)
    dist = 1.0 - cosine_similarity([t1_mean], [t2_mean])[0][0]
    print("Averaged embedding cosine dist:", dist)
    return dist


def compute_jsd_from_cluster_labels(labels1, labels2, name):
    labels_all = list(np.concatenate((labels1, labels2)))
    counts1 = Counter(labels1)
    counts2 = Counter(labels2)
    n_senses = list(set(labels_all))
    print("Clusters:", len(n_senses))

    t1 = np.array([counts1[i] for i in n_senses])
    t2 = np.array([counts2[i] for i in n_senses])

    # compute JS divergence between count vectors by turning them into distributions
    t1_dist = t1 / t1.sum()
    t2_dist = t2 / t2.sum()

    jsd = compute_jsd(t1_dist, t2_dist)
    print("clustering JSD", name+':', jsd)
    return jsd

def compute_wasserstein_distance(labels1, labels2, emb1, emb2, name):
    labels_all = list(np.concatenate((labels1, labels2)))
    counts1 = Counter(labels1)
    counts2 = Counter(labels2)
    n_senses = list(set(labels_all))
    print("Clusters:", len(n_senses))

    t1 = np.array([counts1[i] for i in n_senses])
    t2 = np.array([counts2[i] for i in n_senses])
    t1_dist = t1 / t1.sum()
    t2_dist = t2 / t2.sum()

    emb1_means = np.array([np.mean(emb1[labels1 == clust], 0) for clust in n_senses])
    emb2_means = np.array([np.mean(emb2[labels2 == clust], 0) for clust in n_senses])
    M = np.nan_to_num(np.array([cdist(emb1_means, emb2_means, metric = 'cosine')])[0], nan = 1)
    print("clustering WD", name+':', wass)
    return wass

def detect_meaning_gain_and_loss(labels1, labels2, name):
    all_count = Counter(labels1 + labels2)
    first_count = Counter(labels1)
    second_count = Counter(labels2)
    gained_meaning = False
    lost_meaning = False
    for label, c in all_count.items():
        if c >= 10:
            if label not in first_count or first_count[label] <= 2:
                gained_meaning=True
            if label not in second_count or second_count[label] <= 2:
                lost_meaning=True
    print(name, "gained meaning", gained_meaning, "lost meaning", lost_meaning)
    return str(gained_meaning) + '/' + str(lost_meaning)



def compute_divergence_across_many_periods(embeddings_concat, labels, splits, corpus_slices, name, method):
    all_clusters = []
    clusters_dict = {}
    embs_dict = {}
    all_embs = []
    for split_num, split in enumerate(splits):
        if split_num > 0:
            clusters = list(labels[splits[split_num-1]:split])
            clusters_dict[corpus_slices[split_num - 1]] = clusters
            all_clusters.append(clusters)
            embs = list(embeddings_concat[splits[split_num-1]:split])
            embs_dict[corpus_slices[split_num - 1]] = embs
            all_embs.append(embs)
    all_measures = []
    all_meanings = []
    for i in range(len(all_clusters)):
        if i < len(all_clusters) -1:
            if method == 'JSD':
                dist = compute_jsd_from_cluster_labels(all_clusters[i],all_clusters[i+1], name + ' slice ' + str(i + 1) + ' and ' + str(i + 2))
            elif method == 'WD':
                dist = compute_wasserstein_distance(all_clusters[i],all_clusters[i+1],all_embs[i],all_embs[i+1], name + ' slice ' + str(i + 1) + ' and ' + str(i + 2))
            meaning = detect_meaning_gain_and_loss(all_clusters[i],all_clusters[i+1], name + ' slice ' + str(i + 1) + ' and ' + str(i + 2))
            all_meanings.append(meaning)
            all_measures.append(dist)
    if method == 'JSD':
        entire_measure = compute_jsd_from_cluster_labels(all_clusters[0],all_clusters[-1], name + " First and last slice")
    elif method == 'WD':
        entire_measure = compute_wasserstein_distance(all_clusters[0],all_clusters[-1],all_embs[0],all_embs[-1], name + " First and last slice")
    meaning = detect_meaning_gain_and_loss(all_clusters[0],all_clusters[-1], name + " First and last slice")
    all_meanings.append(meaning)

    avg_measure = sum(all_measures)/len(all_measures)
    all_measures.extend([entire_measure, avg_measure])
    all_measures = [float("{:.6f}".format(score)) for score in all_measures]
    return all_measures, all_meanings, clusters_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure semantic shift')
    parser.add_argument('--task', type=str, default='aylien', help='Name of the corpus to analyse.')
    parser.add_argument('--method', type=str, default='JSD', help='Measure to use: WD or JSD')
    parser.add_argument('--corpus_slices_type', type=str, default='months', help='Name of the corpus to analyse')
    parser.add_argument('--get_additional_info', type=bool, default=True, help='Wether the cluster labels and sentences are saved or not.')
    parser.add_argument('--target_words_path', type=str, default=None, help='Path of a txt file containing the list of target words to analyse. Set to None to use default.')
    parser.add_argument('--results_path', type=str, default='', help='Path to the folder to save the results.')
    parser.add_argument('--emb_path', type=str, help='Path to the embeddings folder.')
    args = parser.parse_args()
    
    emb_path = args.emb_path
    method = args.method
    task = args.task
    get_additional_info = args.get_additional_info
    corpus_slices_type = args.corpus_slices_type
    results_path = args.results_path
    random_state = 123
    print('##############', task, corpus_slices_type, '##############')

    #task='aylien'
    #get_additional_info = True
    if task=='coha':
        results_dir = results_path + "coha_results/"
        corpus_slices = ['1960', '1990']
        embeddings_dict = {
            'english':
                { 'fine_tuned_averaged': 'embeddings/coha_5_yearly_fine_tuned_200_0.99.pickle'},
        }
    elif task=='aylien':
        results_dir = results_path +  "aylien_results/"
        if corpus_slices_type=='months':
            corpus_slices = ['january', 'february', 'march', 'april']
            path = 'embeddings/aylien_monthly_balanced_fine_tuned.pickle'
        elif corpus_slices_type=='sources':
            corpus_slices = ['fox', 'cnn']
            path = 'embeddings/aylien_fox_cnn_balanced_fine_tuned.pickle'
        embeddings_dict = {
            'english':{'fine_tuned_averaged': path}
        }
    elif task=='semeval':
        results_dir = results_path + "semeval_scalable_results/"
        #corpus_slices = ['january', 'february', 'march', 'april']
        corpus_slices = ['1', '2']
        embeddings_dict = {
            'german':
                {'fine_tuned_averaged': 'embeddings/german_5_epochs_scalable.pickle'},
            'english':
                {'fine_tuned_averaged': 'embeddings/english_5_epochs_scalable.pickle'},
            'swedish':
                {'fine_tuned_averaged': 'embeddings/swedish_5_epochs_scalable.pickle'},
            'latin':
                {'fine_tuned_averaged': 'embeddings/latin_5_epochs_scalable.pickle'},

        }
    elif task=='syntetic':
        results_dir = results_path + "syntetic_results/"
        corpus_slices = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']
        embeddings_dict = {
            'english':
                {'fine_tuned_averaged': 'embeddings/syntetic_pretrained.pickle'},
        }
    
    print("Results will be saved at", results_dir)
    target_words = []
    if args.target_words_path is not None:
        file = open(args.target_words_path,"r") 
        target_words = file.read().split('\t')
        file.close()
        print("Target words imported:", len(target_words))
    #target_words = ['economy', 'covid-19']
    
    if get_additional_info and len(target_words) == 0:
        print('Define a list of target words or set "get_additional_info" flag to False')
        sys.exit()

    for lang, configs in embeddings_dict.items():
        for emb_type, embeddings_file in configs.items():
            print("Loading ", embeddings_file)
            try:
                bert_embeddings, count2sents = pickle.load(open(emb_path + embeddings_file, 'rb'))
            except:
                bert_embeddings = pickle.load(open(emb_path + embeddings_file, 'rb'))
                count2sents = None

            #if no predefined list of target words
            if len(target_words) == 0:# or len(target_words) > 10:
                target_words = list(bert_embeddings.keys())

            jsd_vec = []
            cosine_dist_vec = []
            results_dict = {"word": [], "aff_prop": [], "kmeans_5": [], "kmeans_7": [], "averaging": [],
                            "aff_prop_clusters": []}

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


            for i, word in enumerate(target_words):
                #if i > 100:
                #    break
                print("\n=======", i + 1, "- word:", word.upper(), "=======")

                if word not in bert_embeddings:
                    continue
                emb = bert_embeddings[word]
                #print(emb.keys())

                all_embeddings = []
                all_sentences = {}
                splits = [0]
                all_slices_present = True
                all_freqs = []

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


                embeddings_concat = np.concatenate(all_embeddings, axis=0)

                #return no change, i.e., all zeros
                if embeddings_concat.shape[0] < 7 or not all_slices_present:
                    continue
                    #word_results = [word] + ([0] * (len(corpus_slices) + 1) * 3)  + all_freqs + ([0] * (len(corpus_slices)))
                    #print(word_results)
                else:
                    aff_prop_labels, aff_prop_centroids = cluster_word_embeddings_aff_prop(embeddings_concat)
                    all_aff_prop_jsds, all_meanings, clustered_aff_prop_labels = compute_divergence_across_many_periods(embeddings_concat, aff_prop_labels, splits, corpus_slices, 'AFF PROP', method)
                    kmeans_5_labels, kmeans_5_centroids = cluster_word_embeddings_k_means(embeddings_concat, 5, random_state)
                    all_kmeans5_jsds, all_meanings, clustered_kmeans_5_labels = compute_divergence_across_many_periods(embeddings_concat, kmeans_5_labels, splits, corpus_slices, 'KMEANS 5', method)
                    kmeans_7_labels, kmeans_7_centroids = cluster_word_embeddings_k_means(embeddings_concat, 7, random_state)
                    all_kmeans7_jsds, all_meanings, clustered_kmeans_7_labels = compute_divergence_across_many_periods(embeddings_concat, kmeans_7_labels, splits, corpus_slices, 'KMEANS 7', method)
                    all_freqs = all_freqs + [sum(all_freqs)] + [sum(all_freqs)/len(all_freqs)]
                    word_results = [word] +  all_aff_prop_jsds + all_kmeans5_jsds + all_kmeans7_jsds + all_freqs + all_meanings

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
            clust_list = [' AP', ' K5', ' K7']
            methods =[ method + clust for clust in clust_list]+ ['FREQ', 'MEANING GAIN/LOSS']
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


            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            csv_file = results_dir + "results_" + lang + "_" + emb_type + ".csv"


            # save results to CSV
            results_df = pd.DataFrame(results, columns=columns)
            results_df = results_df.sort_values(by=[method + ' K5 Avg'], ascending=False)
            results_df.to_csv(csv_file, sep=';', encoding='utf-8', index=False)

            print("Done! Saved results in", csv_file, "!")

            if get_additional_info:

                # save cluster labels and sentences to pickle
                dicts = [(aff_prop_centroids_dict, 'aff_prop_centroids'), (aff_prop_labels_dict, 'aff_prop_labels'),
                         (kmeans_5_centroids_dict, 'kmeans_5_centroids'), (kmeans_5_labels_dict, 'kmeans_5_labels'),
                         (kmeans_7_centroids_dict, 'kmeans_7_centroids'), (kmeans_7_labels_dict, 'kmeans_7_labels'),
                         (sentence_dict, "sents")]

                for data, name in dicts:
                    data_file = os.path.join(results_dir, name + "_" + lang + "_" + emb_type + ".pkl")
                    centroids_file = results_dir + "aff_prop_centroids_" + lang + "_" + emb_type + ".pkl"
                    pf = open(data_file, 'wb')
                    pickle.dump(data, pf)
                    pf.close()

