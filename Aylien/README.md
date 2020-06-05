
## Lets keep here files related to analysis of Ayliens COVID corpus.

Get Aylien corpus from here: https://aylien.com/coronavirus-news-dataset/

## Static embeddings:

To build embeddings from scratch get Glove embeddings from here: https://nlp.stanford.edu/projects/glove/, then run:
```
python aylien_glove.py <path to aylien-covid-news.jsonl> <path to a Glove model, e.g. glove.42B.300d.txt>'
```

This will take one day or so.

Download ready embeddings from here:
https://www.dropbox.com/s/6a8y6fu4o4n34n2/aylien_glove.zip?dl=0

These are an npy file with embeddings and a pkl with ids.

Read like that:

```python
embeddings = np.load("embeddings_glove.42B.300d.txt.npy")
ids = pickle.load(open("ids_glove.42B.300d.txt.pkl", 'rb'))	
id_to_emb = {doc_id:embeddings[i,:] for i,doc_id in enumerate(ids)}
```


## Time-aware clustering
```
time_aware_clustering.py --data aylien-covid-news.jsonl --embs embeddings_glove.42B.300d.txt.npy --ids ids_glove.42B.300d.txt.pkl 
```
