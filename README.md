# Scalable and Interpretable Semantic Change Detection

Official repository for paper "Scalable and Interpretable Semantic Change Detection" published in Proceedings of NAACL 2021. Published results were produced in Python 3 programming environment on Linux Mint 18 Cinnamon operating system. Instructions for installation assume the usage of PyPI package manager.<br/>


## Installation, documentation ##

Install dependencies if needed: pip install -r requirements.txt

### To reproduce the results published in the paper run the code in the command line using following commands: ###

#### Download all the required data:<br/>

* COHA corpus (https://www.english-corpora.org/coha/), namely texts from 1960 and 1960 periods. A list of gold standard semantic change scores for 100 manually selected words (see https://www.aclweb.org/anthology/W11-2508.pdf) also needs to be acquired.
* SEMEVAL corpora (https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/) together with gold standard annotations for each of the four languages.
* DURel corpus (https://www.ims.uni-stuttgart.de/en/research/resources/corpora/wocc) + gold standard annotations.
* Aylien corpus (https://aylien.com/blog/free-coronavirus-news-dataset), namely articles for January, February, March and April.


#### Prepare the data :<br/> 

Generate COHA language model train and test sets and preprocess the corpus:<br/>

```
python build_coha_corpus.py  --input_folders pathToCOHACorpusSlicesSeparatedBy';' --output_files pathToPreprocessedTxtFilesOnePerEachSliceSeparatedBy';' --lm_output_train pathToOutputLanguageModelTrainFile --lm_output_test pathToOutputLanguageModelTestFile
```

Generate SEMEVAL language model train and test sets for each language and preprocess the corpora:<br/>

```
python build_semeval_lm_train_test.py  --corpus_paths pathToCorpusSlicesSeparatedBy';' --target_path pathToSemEvalTargetFile --language language --lm_train_test_folder pathToOutputFolder
python build_semeval_corpora.py  --corpus_paths pathToCorpusSlicesSeparatedBy';' --target_path pathToSemEvalTargetFile --language language --output_folder pathToOutputFolder
```

Generate DURel language model train and test sets (DURel corpus does not require any preprocessing):<br/>

```
python build_durel_corpus.py  --input_files pathToCorpusSlicesSeparatedBy';' --lm_output_train pathToOutputLanguageModelTrainFile --lm_output_test pathToOutputLanguageModelTestFile
```

Generate Aylien language model train and test sets and preprocess the corpus:<br/>

```
python build_aylien_corpus.py  --input_path pathToAylienJSONFile --output_folder pathToOutputFolderWithTxTFilesForEachSlice --lm_output_train pathToOutputLanguageModelTrainFile --lm_output_test pathToOutputLanguageModelTestFile
```

#### Fine-tune language model:<br/>

Fine-tune BERT model:<br/>

```
python fine-tune_BERT.py --train_data_file pathToLMTrainSet --output_dir pathToOutputModelDir --eval_data_file pathToLMTestSet --model_name_or_path modelForSpecificLanguage --mlm --do_train --do_eval --evaluate_during_training
```

For '--model_name_or_path' parameter, see the paper for info about which models we use for each language. **For SEMEVAL and DURel**, the sentences in the corpora are shuffled, therefore the context is limited to sentences. For this reason **USE AN ADDITIONAL '--line_by_line' flag** when training on this corpora.

#### Extract BERT embeddings:<br/>

Extract embeddings from the preprocessed corpus in .txt for one of the corpora from the SemEval semantic change competiton:<br/>

```
python get_embeddings_scalable_semeval.py --corpus_paths pathToPreprocessedCorpusSlicesSeparatedBy';' --target_path pathToSemEvalTargetFile --language language --path_to_fine_tuned_model pathToFineTunedModel --embeddings_path pathToOutputEmbeddingFile
```

Extract embeddings from the preprocessed corpus in .txt for COHA, DURel or Aylien corpus:<br/>

```
python get_embeddings_scalable.py --corpus_paths pathToPreprocessedCorpusSlicesSeparatedBy';' --target_path pathToTargetFile --task chooseBetween'coha','durel','aylien' --path_to_fine_tuned_model pathToFineTunedModel --embeddings_path pathToOutputEmbeddingFile
```

This creates a pickled file containing all contextual embeddings for all target words.<br/>

#### Get results:<br/>

Conduct clustering and measure semantic shift with various methods:<br/>

```
python measure_semantic_shift.py --task corpusToAnalyse --corpus_slices_type nameOfCopusSlices --emb_path pathToInputEmbeddingFile --results_path pathToOutputResultsDir --method JSD_or_WD
```

This script takes the pickled embedding file as an input and creates several files, a csv file containing semantic change scores for each target word (from the full vocabulary or from a pre-defined list) using either Wasserstein distance or Jensen-Shannon divergence, files containing cluster labels for each embedding , files containing cluster centroids and a file containing context (sentence) mapped to each embedding (optionally).<br/>

Extract keywords and plot clusters distribution for interpretation:<br/>

```
python interpretation_aylien.py  --emb_path pathToInputEmbeddingFile --res_path PathToClusteringResults --save_path PathToSaveClusters
```

This script takes the pickled embedding file and the result file (csv) from the previous step. It automatically selects a set of target words among the most drifting ones (but you can also define your own target list), performs clustering (using measure_semantic_shift.py), extracts keywords for each cluster, and plots the cluster distribution on each corpus slice.


Generate SemEval submission files for task 1 (binary classification using stopword tresholding method) and task 2 (ranking) using a specific clustering method:<br/>

```
python make_semeval_answer_file.py --language language --results_file pathToInputResultsCSVFile --method clusteringMethod --target_path pathToSemEvalTargetFile
```

This script takes the CSV file generated in the previous step as an input and creates SemEval submission files for a specific clustering method (options are 'aff_prop', 'kmeans_5', 'kmeans_7', 'averaging') and language.<br/>

Generate SemEval submission files for task 1 (binary classification) using time period specific cluster method:<br/>

```
python get_period_specific_clusters.py --language language --results_file pathToInputClusterLabelFile --target_path pathToSemEvalTargetFile
```
This script takes one of the cluster labels files generated with the calculate_semantic_change.py script as an input. Use the "--dynamic_treshold" flag if your input labels are for affinity propagation clustering.<br/>

#### Extra:<br/>

Filter Named entities from clusters:<br/>

```
python filter_ner.py --language language --input_sent_file pathToFileWithSentences --input_label_file pathToInputClusterLabelFile --output_dir_path pathToOutputResultsDir
```

This script takes one of the cluster labels files and a sentence file generated with the calculate_semantic_change.py script as an input. It is only appropriate for filtering of affinity propagation clusters.<br/>

Script for ensembling of static word2vec and contextual embeddings:<br/>

```
python ensembling_script.py --language language --method_1 clusteringMethodName --input_file_method_1 pathToInputResultsCSVFile --method_2 word2vecMethodName --input_file_method_2 pathToInputWord2VecFile --output_file_path OutputCSVResultsFile
```

This script takes the CSV file generated with the calculate_semantic_change.py and the name of the column (results for clustering) in the CSV file as an input, and also the CSV file generated with the code generated
by using the code published at https://github.com/Garrafao/LSCDetection (see their readme on how to produce the CSV file with the measures/cd.py script) and the name of the column in that CSV file. 

#### If something is unclear, check the default arguments for each script. If you still can't make it work, feel free to contact us :).

