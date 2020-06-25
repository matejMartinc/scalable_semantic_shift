import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
import random
from scipy import sparse
import argparse
import itertools
from scipy.io import savemat, loadmat
import spacy
spacy_nlp = spacy.load('en_core_web_sm')
# python -m spacy download en
nlp = spacy.load('en')
import nltk
import glob
import gensim
from gensim.utils import simple_preprocess
import os, re
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import time
from scipy.stats import entropy
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
import subprocess

# import evaluation measures
from evaluation_scripts.evaluate_clustering_keywords import *



def select_keywords(results_df, method='AP', frequency_threshold=30, nb_words = 50):
    """
    :param results_df: the dataframe of drifts results for a given dataset.
    :param method: either AP, K5 or K5 (aff-prop or kmeans)
    :return: list of target words to analyse.
    It also saves the list of target words to a txt file named "aylien_target_list.txt".
    """
    results_df_changed = results_df[results_df['MEANING GAIN/LOSS All'].str.contains("True")]
    freq_cols = [col for col in results_df if col.startswith('FREQ')]
    for col in freq_cols:
        results_df_changed = results_df_changed[(results_df_changed[col] > frequency_threshold)]
    target_res = results_df_changed.sort_values(by='JSD '+ method +' Avg', ascending = False)[:nb_words]
    target_words = list(target_res['word'])
    target_words_reduc = [word for word in target_words if len(word)>1]
    print(len(results_df),len(results_df_changed), len(target_res), len(target_words_reduc))
    file = open(target_words_path,"w+") 
    file.write("\t".join(target_words_reduc))
    file.close()
    return target_words_reduc

def get_clusters_sent(target, method, corpus_slices_type, threshold_size_cluster):
    #sentences_dict = defaultdict(lambda : defaultdict(list))
    
    labels = "aylien_results/" + method + "_labels_english_fine_tuned_averaged.pkl"
    sentences = "aylien_results/sents_english_fine_tuned_averaged.pkl"
    corpus_slices = []
    if corpus_slices_type=='months':
        corpus_slices = ['january', 'february', 'march', 'april']
    elif corpus_slices_type=='sources':
        corpus_slices = ['fox', 'cnn']
    else:
        print("corpus_slices_type argument not recognised")

    labels = pickle.load(open(labels, 'rb'))
    print(labels.keys())
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
    sentences = []
    labels = []
    categs = []

    for label, count in sorted_counts:
        print("\n================================")
        print("Cluster label: ", label, " - Cluster size: ", count)
        if count>threshold_size_cluster:
            for cs in corpus_slices:
                print("Corpus slice: ", cs, " - Num appearances: ", counts[cs].get(label, 0))
                for s in cluster_to_sentence[label][cs]:
                    sent_clean = s.replace("[CLS]", "").replace("[SEP]", "").strip()
                    #sentences_dict[label][cs].append(sent_clean)
                    sentences.append(sent_clean)
                    labels.append(label)
                    categs.append(cs)
        else:
            print("Cluster", label, "is too small - deleted!")
    
    sent_df = pd.DataFrame(list(zip(sentences, labels, categs)), columns =['sentence', 'label', 'categ'])
    
    return sent_df

def output_distrib(data, word):
    k = len(data['label'].unique())

    distrib = data.groupby(['categ', "label"]).size().reset_index(name="count")
    pivot_distrib = distrib.pivot(index='categ', columns='label', values='count')
    pivot_distrib_norm = pivot_distrib.div(pivot_distrib.sum(axis=1), axis=0)
    pivot_distrib_norm.columns = [str(i) for i in list(range(k))]

    pivot_distrib_norm.plot.bar(stacked=True, title="Word: \"" + word + '\"', colormap='Spectral')
    return pivot_distrib_norm


def extract_topn_from_vector(feature_names, sorted_items, topn):
    """get the feature names and tf-idf score of top n items"""
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]   
    return results

def extract_keywords(word_clustered_data, max_df, topn, lemmatisation):
    
    # get groups of sentences for each cluster
    sent_clust_dict = dict()
    for i, row in word_clustered_data.iterrows():
        if row['label'] in sent_clust_dict:
            sent_clust_dict[row['label']] = sent_clust_dict[row['label']] + ' ' + row['sentence']
        else:
            sent_clust_dict[row['label']] = row['sentence']
    sent_clust_dict.keys()
    
    stop1 = list(spacy.lang.en.stop_words.STOP_WORDS)
    stop2 = stopwords.words('english')
    stop = set(stop1 + stop2)
    #print(len(stop1), len(stop2), len(stop))
    
    cv=CountVectorizer(max_df = max_df, stop_words=stop, max_features=10000) 
    # max_df ignore all words that have appeared in X % of the documents
    # max_features is the vocab size
    labels, clusters = list(sent_clust_dict.keys()), list(sent_clust_dict.values())
    
    #no_lone=' '.join( [w for w in doc.split() if len(w)>1] ) # also remove multiple spaces    
    if lemmatisation:
        sentences_clusters_split = [[token.lemma_ for token in nlp(doc)] for doc in clusters]
        clusters = [' '.join(doc_split) for doc_split in sentences_clusters_split]
    else:
        sentences_clusters_split = [doc.split() for doc in clusters]
    
    word_count_vector=cv.fit_transform(clusters)
    #print(word_count_vector.shape)
    feature_names=cv.get_feature_names()

    #print(list(cv.vocabulary_.keys())[:10])
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    keyword_clusters = {}
    for label, cluster in zip(labels, clusters):
        #generate tf-idf 
        tf_idf_vector=tfidf_transformer.transform(cv.transform([cluster]))
        #sort the tf-idf vectors by descending order of scores
        tuples = zip(tf_idf_vector.tocoo().col, tf_idf_vector.tocoo().data)
        sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
        #extract only the top n
        keywords=extract_topn_from_vector(feature_names, sorted_items, topn)
        
        keyword_clusters[label] = keywords

    return keyword_clusters, sentences_clusters_split



def full_analysis(word, max_df = 0.7 , topn=15, method = "kmeans_5", lemmatisation = True, corpus_slices_type='months', threshold_size_cluster=10):
    clusters_sents_df = get_clusters_sent(word, method, corpus_slices_type, threshold_size_cluster)
    pivot_distrib = output_distrib(clusters_sents_df, word)
    keyword_clusters, sentences_clusters = extract_keywords(clusters_sents_df, topn = topn, max_df = max_df, lemmatisation = lemmatisation)
    for k in keyword_clusters:
        print(k)
        print(list(keyword_clusters[k].keys()))
    evaluate_clustering(keyword_clusters, sentences_clusters)
    return keyword_clusters



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure semantic shift')
    parser.add_argument('--emb_path', type=str, help='Path to the embeddings (pickel file).')
    parser.add_argument('--res_path', type=str, help='Path to the results of the clustering (csv file).')

    parser.add_argument('--method', type=str, default='K5', help='Method to use to select the results for analysis: K5, K7 or AP.')
    parser.add_argument('--nb_target', type=int, default='5', help='Number of target words to select')
    parser.add_argument('--save_path', type=str, default='', help='Path to save the results.')
    args = parser.parse_args()
    
    if args.method == 'K7':
        method = 'kmeans_7'
    elif args.method == 'K5':
        method = 'kmeans_5'
    elif args.method == 'AP':
        method = 'aff_prop'
    target_words_path = 'aylien_target_list.txt'
    
    print("Loading results")
    #aylien_emb, aylien_count2sents = pickle.load(open(args.emb_path, 'rb'))
    res_aylien = pd.read_csv(args.res_path, sep=';')

    if 'fox_cnn' in args.res_path:
        categ = "sources"
    elif 'monthly' in args.res_path:
        categ = "months"
    
    print("Selecting keywords:")
    target_words_reduc = select_keywords(results_df=res_aylien, method=args.method, frequency_threshold=60, nb_words = args.nb_target)
    print(target_words_reduc)
    
    print("Extracting results")
    #os.system('python measure_semantic_shift.py --corpus_slices_type categ --results_path args.save_path --target_words_path target_words_path')
    subprocess.call(["python", "measure_semantic_shift.py", "--corpus_slices_type", categ, '--results_path', args.save_path , '--target_words_path', target_words_path, '--emb_path', args.emb_path])
    
    for word in target_words_reduc:
        print('##########################################', word, '################################')
        keyword_clusters = full_analysis(word = word, method = method, topn=25, lemmatisation = True, corpus_slices_type = categ)
    
    
    
    
    # exemple of how to use it:
    #python interpretation_aylien.py --res_path '/people/montariol/data/scalable_semantic_change/results/aylien_fox_cnn_results_english_fine_tuned_averaged.csv' --emb_path '/people/montariol/data/scalable_semantic_change/' --nb_target 5
    
    
    
    
    
    
    
    
    
    
    
    
    
    
