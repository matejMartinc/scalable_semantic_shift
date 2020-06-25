# from https://github.com/MilaNLProc/contextualized-topic-models/blob/master/contextualized_topic_models/evaluation/measures.py
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
import itertools


class Measure:
    def __init__(self):
        pass
    def score(self):
        pass


class TopicDiversity(Measure):
    def __init__(self, topics):
        super().__init__()
        self.topics = topics

    def score(self, topk=25):
        """
        :param topk: topk words on which the topic diversity will be computed
        :return:
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk:', len(self.topics[0]))
        else:
            unique_words = set()
            for t in self.topics:
                unique_words = unique_words.union(set(t[:topk]))
            td = len(unique_words) / (topk * len(self.topics))
            return td


class CoherenceNPMI(Measure):
    def __init__(self, topics, texts):
        super().__init__()
        self.topics = topics
        self.texts = texts
        self.dictionary = Dictionary(self.texts)

    def score(self, topk=10):
        """
        :param topics: a list of lists of the top-k words
        :param texts: (list of lists of strings) represents the corpus on which the empirical frequencies of words are
        computed
        :param topk: how many most likely words to consider in the evaluation
        :return:
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk:', len(self.topics[0]))
        else:
            npmi = CoherenceModel(topics=self.topics, texts=self.texts, dictionary=self.dictionary,
                                  coherence='c_npmi', topn=topk)
            return npmi.get_coherence()


class CoherenceWordEmbeddings(Measure):
    def __init__(self, topics, word2vec_path=None, binary=False):
        '''
        :param topics: a list of lists of the top-n most likely words
        :param word2vec_path: if word2vec_file is specified, it retrieves the word embeddings file (in word2vec format) to
         compute similarities between words, otherwise 'word2vec-google-news-300' is downloaded
        :param binary: if the word2vec file is binary
        '''
        super().__init__()
        self.topics = topics
        self.binary = binary
        if word2vec_path is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = KeyedVectors.load_word2vec_format(word2vec_path, binary=binary)

    def score(self, topk=10, binary= False):
        """
        :param topk: how many most likely words to consider in the evaluation
        :return: topic coherence computed on the word embeddings similarities
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk:', len(self.topics[0]))
        else:
            arrays = []
            for index, topic in enumerate(self.topics):
                if len(topic) > 0:
                    local_simi = []
                    for word1, word2 in itertools.combinations(topic[0:topk], 2):
                        if word1 in self.wv.vocab and word2 in self.wv.vocab:
                            local_simi.append(self.wv.similarity(word1, word2))
                    arrays.append(np.mean(local_simi))
            return np.mean(arrays)


def evaluate_clustering(keyword_clusters, sentences_clusters):
    """
    :param keyword_clusters: dict of cluster label avec top keywords.
    :param sentences_clusters: list of docs, each doc is split into words.
    :return: list of the 3 scores topic diversity, nmpi, coherence_w2v
    """
    top_keywords = [list(dict_k.keys()) for dict_k in list(keyword_clusters.values())]
    td = TopicDiversity(top_keywords)
    print(td.score(topk=15))
    npmi = CoherenceNPMI(top_keywords, sentences_clusters)
    print(npmi.score(topk=10))
    #w2v = CoherenceWordEmbeddings(top_keywords, word2vec_path=w2v_path, binary=True)
    #print(w2v.score(topk=10))
    results = [td, npmi]
    return results

