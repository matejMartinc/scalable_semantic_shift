import os, sys
import json
import re
from progress.bar import Bar
import pickle
import numpy as np

from collections import defaultdict, Counter

import spacy
nlp = spacy.load("en_core_web_sm")


def readin_embeddings(embeddings_path):
    embeddings_dict = {}
    bar = Bar("Reading embeddings", suffix='%(index).d/%(elapsed).d')
    with open(embeddings_path, 'r') as f:
        for line in f.readlines():
            bar.next()
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    bar.finish()
    return embeddings_dict


def make_document_embeddings(token_counts, idf, embeddings):
    doc_embeddings = None
    total = 0
    for t in token_counts:
        if t in embeddings:
            tf_idf = token_counts[t]*idf[t]
            total += tf_idf
            if doc_embeddings is not None:
                doc_embeddings = doc_embeddings + tf_idf * embeddings[t]
            else:
                doc_embeddings = tf_idf * embeddings[t]
    if total != 0:
        return doc_embeddings / total



if __name__ == "__main__":
    input_file = sys.argv[1]
    embeddings_path = sys.argv[2]

    documents = {}
    df = defaultdict(int)

    bar = Bar("Reading documents")

    with open(input_file) as inp:
        while True: 
            line = inp.readline() 
            if not line: 
                break

            bar.next()
            document = json.loads(line)
            token_counts = Counter([t.lemma_ for t in nlp(document["body"]) if  t.is_alpha and not t.is_stop])
            documents[document["id"]] = token_counts
            for t in token_counts:
                df[t] += token_counts[t]

#            if len(documents)>=200:
#                break
    bar.finish()

    total = len(documents)
    idf = {t:np.log(total/df[t]) for t in df}
    del df

    embeddings_dict = readin_embeddings(embeddings_path)
    
    ids = []
    all_embeddings = None
    bar = Bar("Making embeddings", max=len(documents), suffix = '%(percent).1f%% - %(eta)ds')    
    for doc_id, token_counts in documents.items():
        bar.next()
        doc_embeddings = make_document_embeddings(token_counts, idf, embeddings_dict)
        if doc_embeddings is not None:
            ids.append(doc_id)
            if all_embeddings is not None:
                all_embeddings = np.vstack((all_embeddings, doc_embeddings))
            else:
                all_embeddings = doc_embeddings
    bar.finish()

    np.save("embeddings_"+os.path.basename(embeddings_path), all_embeddings)
    pickle.dump(ids, open("ids_" + os.path.basename(embeddings_path) + ".pkl", "wb"))

