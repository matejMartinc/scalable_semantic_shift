import pickle
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from collections import Counter
from scipy.stats import entropy
import numpy as np
import os



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


def compute_divergence_across_many_periods(labels, splits, corpus_slices):
    all_clusters = []
    clusters_dict = {}
    for split_num, split in enumerate(splits):
        if split_num > 0:
            clusters = list(labels[splits[split_num-1]:split])
            clusters_dict[corpus_slices[split_num - 1]] = clusters
            all_clusters.append(clusters)
    all_jsds = []
    for i in range(len(all_clusters)):
        if i < len(all_clusters) -1:
            jsd = compute_divergence_from_cluster_labels(all_clusters[i],all_clusters[i+1])
            all_jsds.append(jsd)
    return all_jsds, clusters_dict


if __name__ == '__main__':

    coha=False
    get_additional_info = True
    if coha:
        results_dir = "coha_results/"
        corpus_slices = ['1960', '1990']
        embeddings_dict = {
            'english':
                { 'fine_tuned_averaged': 'embeddings/coha_5_yearly_fine_tuned.pickle'},
        }
    else:
        results_dir = "aylien_results/"
        #corpus_slices = ['january', 'february', 'march', 'april']
        corpus_slices = ['fox', 'cnn']
        embeddings_dict = {
            'english':
                #{'fine_tuned_averaged': 'embeddings/aylien_5_monthly_fine_tuned_balanced.pickle'},
                {'fine_tuned_averaged': 'embeddings/aylien_5_cnn_fox_fine_tuned.pickle'}
        }
    #target_words = []
    target_words = ['economy']

    for lang, configs in embeddings_dict.items():
        for emb_type, embeddings_file in configs.items():
            print("Loading ", embeddings_file)

            bert_embeddings, count2sents = pickle.load(open(embeddings_file, 'rb'))

            #if no predefined list of target words
            if len(target_words) == 0:
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

                all_embeddings = []
                all_sentences = {}
                splits = [0]
                all_slices_present = True

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
                    cs_text = cs + '_text'
                    print("Slice: ", cs)
                    print("Num embeds: ", len(emb[cs]))
                    num_sent_codes = 0


                    for idx in range(len(emb[cs])):

                        #get summed embedding and its count, devide embedding by count
                        e, count_emb = emb[cs][idx]
                        e = e/count_emb

                        try:
                            sent_codes = emb[cs_text][idx]
                        except:
                            #unfinished sentence (missing context), ignore this embedding
                            print('Should not happen')
                            continue
                        sents = []
                        num_sent_codes += len(sent_codes)
                        #print("Num sentences: ", len(sent_codes))
                        for sent in sent_codes:
                            if sent in count2sents[cs]:
                                text = count2sents[cs][sent]
                            sents.append(text)
                            #print(text)
                        print(sents)


                        cs_embeddings.append(e)
                        cs_sentences.append(" ".join(sents))

                    all_embeddings.append(np.array(cs_embeddings))
                    all_sentences[cs] = cs_sentences
                    splits.append(splits[-1] + len(cs_embeddings))


                print("Num sents: ", num_sent_codes)


                embeddings_concat = np.concatenate(all_embeddings, axis=0)
                if embeddings_concat.shape[0] < 7 or not all_slices_present:
                    continue


                aff_prop_labels, aff_prop_centroids = cluster_word_embeddings_aff_prop(embeddings_concat)
                all_aff_prop_jsds, clustered_aff_prop_labels = compute_divergence_across_many_periods(aff_prop_labels, splits, corpus_slices)
                kmeans_5_labels, kmeans_5_centroids = cluster_word_embeddings_k_means(embeddings_concat, k=5)
                all_kmeans5_jsds, clustered_kmeans_5_labels = compute_divergence_across_many_periods(kmeans_5_labels, splits, corpus_slices)
                kmeans_7_labels, kmeans_7_centroids = cluster_word_embeddings_k_means(embeddings_concat, k=7)
                all_kmeans7_jsds, clustered_kmeans_7_labels = compute_divergence_across_many_periods(kmeans_7_labels, splits, corpus_slices)
                all_aff_prop_jsds = all_aff_prop_jsds
                all_aff_prop_jsds = all_aff_prop_jsds + [sum(all_aff_prop_jsds)/len(all_aff_prop_jsds)]
                all_kmeans5_jsds = all_kmeans5_jsds + [sum(all_kmeans5_jsds) / len(all_kmeans5_jsds)]
                all_kmeans7_jsds = all_kmeans7_jsds + [sum(all_kmeans7_jsds) / len(all_kmeans7_jsds)]
                word_results = [word] +  all_aff_prop_jsds + all_kmeans5_jsds + all_kmeans7_jsds
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
            methods = ['JSD AP', 'JSD K5', 'JSD K7']
            for method in methods:
                for num_slice, cs in enumerate(corpus_slices):
                    if num_slice < len(corpus_slices) - 1:
                        columns.append(method + ' ' + cs + '-' + corpus_slices[num_slice + 1])
                columns.append(method + ' Avg')


            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            csv_file = results_dir + "results_" + lang + "_" + emb_type + ".csv"


            # save results to CSV
            results_df = pd.DataFrame(results, columns=columns)
            results_df = results_df.sort_values(by=['JSD K5 Avg'], ascending=False)
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




#massie









