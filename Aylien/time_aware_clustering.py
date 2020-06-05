import argparse
import numpy as np
import json, pickle

from sklearn.metrics.pairwise import cosine_similarity

from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data', help="Path to json file", required=True)
parser.add_argument('--embs', help="Embeddings file path", required=True)
parser.add_argument('--ids', help="Ids file path", required=True)

DEBUG = True
debug_use = 100000

def period_distance(array1, array2):
    return  np.mean(cosine_similarity(array1, array2))

def distance_at_place(periods, place, date_to_embs):
    try:
    
        p1_emb = np.vstack([date_to_embs[d] for d in periods[place]])
        p2_emb = np.vstack([date_to_embs[d] for d in periods[place+1]])

    except Exception as e:
        print("ERROR IN distance_at_place")
        print(periods)
        print(place)
        raise e
    return period_distance(p1_emb, p2_emb)


def vizualize(cluster_table, periods):
    plt.title('Time-aware Clustering Dendrogram')
    plt.xlabel('Date')
    plt.ylabel('distance')

    dendrogram(cluster_table, leaf_font_size=8, leaf_rotation=45, labels=periods)

    plt.savefig("dendrogram.png")

def time_aware_clustering(periods, date_to_embs):
    cluster_number = {p:i for i,p in enumerate(periods)}
    cluster_counter = len(periods)
    
    dists = [distance_at_place(periods, i, date_to_embs) for i in range(len(periods)-1)]
    cluster_table = None
    
    while len(periods) > 1:
        print("")
        print(periods)
        print(dists)
        print(cluster_table)
        print("")
        merge_left_index = np.argmax(np.array(dists))

        new_period = periods[merge_left_index]+periods[merge_left_index+1]
        
        new_cluster = [cluster_number[periods[merge_left_index]],
                       cluster_number[periods[merge_left_index+1]],
                       1-dists[merge_left_index],
                       len(new_period)]

        cluster_number[new_period] = cluster_counter
        cluster_counter += 1
        
        if cluster_table is None:
            cluster_table = np.array([new_cluster])
        else:
            cluster_table = np.vstack([cluster_table, np.array(new_cluster)])

        periods = periods[:merge_left_index] + [new_period] + periods[merge_left_index+2:]
        if len(periods) == 1:
            break
        
        new_dists = []
        for i,d in enumerate(dists):
            if i == merge_left_index+1:
                continue
            elif i in [merge_left_index-1, merge_left_index]:
                if i < len(periods)-1:
                    new_dists.append(distance_at_place(periods, i, date_to_embs))
            else:
                new_dists.append(d)
        dists = new_dists
                
            
    print("FINAL: ", cluster_table) 

    return cluster_table
    

if __name__ == "__main__":
    args=parser.parse_args()

    embeddings = np.load(args.embs)
    ids = pickle.load(open(args.ids, 'rb'))

    if DEBUG:
        embeddings = embeddings[:debug_use,:]
        ids = ids[:debug_use]

    id_to_emb = {doc_id:embeddings[i,:] for i,doc_id in enumerate(ids)}
    del embeddings
    del ids
    
    date_to_embs = {}
    d = 0
    with open(args.data) as dat:
        while True:
            line = dat.readline()
            if not line:
                break
            document = json.loads(line)
            doc_id = document["id"]
            if not doc_id  in id_to_emb:
                continue
                        
            d+=1
            
            date = document["published_at"].split()[0]

            if date not in date_to_embs:
                date_to_embs[date] = id_to_emb[doc_id]
            else:
                date_to_embs[date] = np.vstack((date_to_embs[date], id_to_emb[doc_id]))
                        
            if DEBUG and d==debug_use:
                print(sorted(date_to_embs.keys()))
                break

    periods=[(d,) for d in sorted(date_to_embs.keys())]
    cluster_table = time_aware_clustering(periods, date_to_embs)
    np.save("cluster_table", cluster_table)
    vizualize(cluster_table, periods)
    
