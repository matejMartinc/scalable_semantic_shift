

## Instructions ##


Install dependencies if needed:<br/>
pip install -r requirements.txt<br/><br/>
Add coha corpus (folders COHA_1960 and COHA_19900) and Gulordava_word_meaning_change_evaluation_dataset.csv into data/coha/.<br/>
Run 'python build_coha_corpus'.<br/>
This generates two .txt files, one for each time period, which are used  as input to embedding extraction script.<br/><br/>
If you are using fine-tuned BERT, change the path to the model in get_embeddings_scalable.py accordingly. Otherwise, comment the 'state_dict' line. <br/>
Run 'python get_embeddings_scalable.py' to extract coha, syntetic or aylien embeddings. <br/>
Run 'python get_embeddings_scalable_semeval.py' to extract semeval embeddings. <br/>
Run python measure_semantic_shift.py to cluster embeddings and either make a ranked list or extract and save clusters and its sentences (same as in the semeval code) for some predefined target words.<br/>
Run get_cluster_sents.py to extract sentences from the cluster sentence object generated in the previous step <br/>
Run generate_syntetic_gs.py to make a gold standard list for syntetic dataset <br/>
Folder evluation_scripts contains scripts that calculate pearson coeficients. Inputs are ranked list and gold standard ranked list.<br/>
Folder make_dataset_scripts contains scripts that generate the input coha coprus and two aylien corpora. You can also get some stats about the corpora.<br/>



