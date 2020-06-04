import pickle
import pandas as pd
import re

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from collections import Counter
from scipy.stats import entropy
import numpy as np
import os
from transformers import BertTokenizer


def compute_jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def create_sent2pos_sent(paths, target_path):
    dict = {}
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    counter = 0
    targets = []
    with open(target_path, 'r', encoding='utf8') as f:
        for line in f:
            target = line.strip()
            if len(target) > 0:
                targets.append(target)
    for path in paths:
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                contains = False
                counter += 1
                if counter % 10000 == 0:
                    print('Tokenizing sent. ', counter)
                clean_line = line
                for w in targets:
                    if w.strip() in set(line.split()):
                        contains = True
                        clean_line = clean_line.replace(w.strip(), w.strip()[:-3])

                if contains:
                    marked_text = "[CLS] " + clean_line + " [SEP]"
                    tokenized_text = tokenizer.tokenize(marked_text)
                    for i in range(0, len(tokenized_text), 256):
                        text = tokenized_text[i:i + 256]
                        text = " ".join(text).replace(" ##", "")
                        dict[text] = line
    return dict

def filter_english(text, word):
    if word in text and word[:-3] not in text.split():
        return False
    else:
        print(text, word)
        return True



def cluster_word_embeddings_aff_prop(word_embeddings, preference=None):
    if preference is not None:
        clustering = AffinityPropagation(preference=preference).fit(word_embeddings)
    else:
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


def cluster_word_embeddings_k_means(word_embeddings, k=3):
    clustering = KMeans(n_clusters=k, random_state=0).fit(word_embeddings)
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


def compute_divergence_from_cluster_labels(labels1, labels2):
    labels_all = list(np.concatenate((labels1, labels2)))
    counts1 = Counter(labels1)
    counts2 = Counter(labels2)
    n_senses = list(set(labels_all))
    # print("Clusters:", len(n_senses))

    t1 = np.array([counts1[i] for i in n_senses])
    t2 = np.array([counts2[i] for i in n_senses])

    # compute JS divergence between count vectors by turning them into distributions
    t1_dist = t1 / t1.sum()
    t2_dist = t2 / t2.sum()

    jsd = compute_jsd(t1_dist, t2_dist)
    print("clustering JSD:", jsd)
    return jsd


if __name__ == '__main__':

    kmeans_clustering = False
    embeddings_dict = {
        'english':
            {   'fine_tuned_averaged': 'embeddings/coha_5_yearly_fine_tuned.pickle',
            },
    }


    for lang, configs in embeddings_dict.items():
        for emb_type, embeddings_file in configs.items():

            bert_embeddings, count2sents = pickle.load(open(embeddings_file, 'rb'))

            #with open('test.pickle', 'wb') as handle:
            #    pickle.dump(bert_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

            target_words = list(bert_embeddings.keys())
            print(target_words)

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
            for i, word in enumerate(target_words):
                print("\n=======", i + 1, "- word:", word.upper(), "=======")
                emb = bert_embeddings[word]

                embeddings1 = []
                embeddings2 = []
                texts1 = []
                texts2 = []

                regex = r"\b%s\b" % word.replace("_vb", "").replace("_nn", "")

                time_slices = ['t1960', 't1990']

                for ts in time_slices:
                    count_all = 0
                    avg_embed = np.zeros(768)

                    text_seen = set()

                    counts = [x[1] for x in emb[ts]]
                    #print(counts)

                    for idx in range(len(emb[ts])):
                        ts_text = ts + '_text'
                        e, count_emb = emb[ts][idx]
                        e = e/count_emb
                        if not kmeans_clustering:
                            try:
                                sent_codes = emb[ts_text][idx]
                            except:
                                #unfinished sentence (missing context), ignore this embedding
                                continue

                            sents = []
                            for sent in sent_codes:
                                if sent in count2sents[ts]:
                                    text = count2sents[ts][sent]
                                sents.append(text)

                        if ts == time_slices[0]:
                            #if count_emb > 1:
                            embeddings1.append(e)
                            if not kmeans_clustering:
                                texts1.append(sents)
                        elif ts == time_slices[1]:
                            #if count_emb > 1:
                            embeddings2.append(e)
                            if not kmeans_clustering:
                                texts2.append(sents)

                embeddings1 = np.array(embeddings1)
                embeddings2 = np.array(embeddings2)

                print("t1 num. occurences: ", embeddings1.shape[0])
                print("t2 num. occurences: ", embeddings2.shape[0])

                sentence_dict[word] = {time_slices[0]: texts1, time_slices[1]: texts2}

                average_dist = compute_averaged_embedding_dist(embeddings1, embeddings2)

                embeddings_concat = np.concatenate([embeddings1, embeddings2], axis=0)

                aff_prop_labels, aff_prop_centroids = cluster_word_embeddings_aff_prop(embeddings_concat)
                clusters1_aff = list(aff_prop_labels[:embeddings1.shape[0]])
                clusters2_aff = list(aff_prop_labels[embeddings1.shape[0]:])
                n_senses = len(list(set(aff_prop_labels)))
                aff_prop_jsd = compute_divergence_from_cluster_labels(clusters1_aff, clusters2_aff)

                kmeans_5_labels, kmeans_5_centroids = cluster_word_embeddings_k_means(embeddings_concat, k=5)
                clusters1_km5 = list(kmeans_5_labels[:embeddings1.shape[0]])
                clusters2_km5 = list(kmeans_5_labels[embeddings1.shape[0]:])
                kmeans5_jsd = compute_divergence_from_cluster_labels(clusters1_km5, clusters2_km5)

                kmeans_7_labels, kmeans_7_centroids = cluster_word_embeddings_k_means(embeddings_concat, k=7)
                clusters1_km7 = list(kmeans_7_labels[:embeddings1.shape[0]])
                clusters2_km7 = list(kmeans_7_labels[embeddings1.shape[0]:])
                kmeans7_jsd = compute_divergence_from_cluster_labels(clusters1_km7, clusters2_km7)

                # add results to dataframe for saving
                aff_prop_labels_dict[word] = {time_slices[0]: clusters1_aff, time_slices[1]: clusters2_aff}
                aff_prop_centroids_dict[word] = aff_prop_centroids

                kmeans_5_labels_dict[word] = {time_slices[0]: clusters1_km5, time_slices[1]: clusters2_km5}
                kmeans_5_centroids_dict[word] = kmeans_5_centroids

                kmeans_7_labels_dict[word] = {time_slices[0]: clusters1_km7, time_slices[1]: clusters2_km7}
                kmeans_7_centroids_dict[word] = kmeans_7_centroids  # add results to dataframe for saving

                results_dict["word"].append(word)
                results_dict["aff_prop"].append(aff_prop_jsd)
                results_dict["aff_prop_clusters"].append(n_senses)
                results_dict["kmeans_5"].append(kmeans5_jsd)
                results_dict["kmeans_7"].append(kmeans7_jsd)
                results_dict["averaging"].append(average_dist)

            # save everything
            results_dir = "coha_results/"
            if not os.path.exists(results_dir):
                os.makedirs("coha_results/")

            csv_file = results_dir + "results_" + lang + "_" + emb_type + ".csv"
            labels_file = results_dir + "labels_" + lang + "_" + emb_type + ".pkl"
            centroids_file = results_dir + "centroids_" + lang + "_" + emb_type + ".pkl"
            sents_file = results_dir + "sents_" + lang + "_" + emb_type + ".pkl"

            # save results to CSV
            results_df = pd.DataFrame.from_dict(results_dict)
            results_df = results_df.sort_values(by=['aff_prop'], ascending=False)
            results_df.to_csv(csv_file, sep='\t', encoding='utf-8', index=False)

            # save cluster labels to pickle
            labels_file = results_dir + "aff_prop_labels_" + lang + "_" + emb_type + ".pkl"
            centroids_file = results_dir + "aff_prop_centroids_" + lang + "_" + emb_type + ".pkl"
            pf = open(labels_file, 'wb')
            pickle.dump(aff_prop_labels_dict, pf)
            pf.close()
            pf2 = open(centroids_file, 'wb')
            pickle.dump(aff_prop_centroids_dict, pf2)
            pf2.close()

            labels_file = results_dir + "kmeans_5_labels_" + lang + "_" + emb_type + ".pkl"
            centroids_file = results_dir + "kmeans_5_centroids_" + lang + "_" + emb_type + ".pkl"
            pf = open(labels_file, 'wb')
            pickle.dump(kmeans_5_labels_dict, pf)
            pf.close()
            pf2 = open(centroids_file, 'wb')
            pickle.dump(kmeans_5_centroids_dict, pf2)
            pf2.close()

            labels_file = results_dir + "kmeans_7_labels_" + lang + "_" + emb_type + ".pkl"
            centroids_file = results_dir + "kmeans_7_centroids_" + lang + "_" + emb_type + ".pkl"
            pf = open(labels_file, 'wb')
            pickle.dump(kmeans_7_labels_dict, pf)
            pf.close()
            pf2 = open(centroids_file, 'wb')
            pickle.dump(kmeans_7_centroids_dict, pf2)
            pf2.close()

            # save sentences
            pf3 = open(sents_file, 'wb')
            pickle.dump(sentence_dict, pf3)
            pf3.close()

            print("Done! Saved results in", csv_file, "!")









