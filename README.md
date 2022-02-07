# Scalable and Interpretable Semantic Change Detection

Official repository for paper "[Scalable and Interpretable Semantic Change Detection](https://aclanthology.org/2021.naacl-main.369/)" published in Proceedings of NAACL 2021. Published results were produced in Python 3 programming environment on Linux Mint 18 Cinnamon operating system. Instructions for installation assume the usage of PyPI package manager.<br/>


## Installation, documentation ##

Install dependencies if needed: pip install -r requirements.txt <br/>
You also need to download 'tokenizers/punkt/english.pickle' using nltk library.

### To reproduce the results published in the paper run the code in the command line using following commands: ###

#### Download all the required data:<br/>

* COHA corpus (https://www.english-corpora.org/coha/), namely texts from 1960 and 1960 periods. A list of gold standard semantic change scores for 100 manually selected words (see https://www.aclweb.org/anthology/W11-2508.pdf) also needs to be acquired.
* SEMEVAL corpora (https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/) together with gold standard annotations for each of the four languages.
* DURel corpus (https://www.ims.uni-stuttgart.de/en/research/resources/corpora/wocc) + gold standard annotations.
* Aylien corpus (https://aylien.com/blog/free-coronavirus-news-dataset), namely articles for January, February, March and April. The list of target words (i.e., vocabulary for which we generate embeddings) can be found in data/aylien folder in the repository.


#### Prepare the data :<br/> 

Generate COHA language model train and test sets and preprocess the corpus:<br/>

```
python build_coha_corpus.py  --input_folders pathToCOHACorpusSlicesSeparatedBy';' --output_files pathToPreprocessedTxtFilesOnePerEachSliceSeparatedBy';' --lm_output_train pathToOutputLanguageModelTrainFile --lm_output_test pathToOutputLanguageModelTestFile
```


**Don't forget to put quotes around --input_folders argument, or ';' will be interpreted as a new command. :)**


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

Fine-tune BERT model for 5 epochs:<br/>

```
python fine-tune_BERT.py --train_data_file pathToLMTrainSet --output_dir pathToOutputModelDir --eval_data_file pathToLMTestSet --model_name_or_path modelForSpecificLanguage --mlm --do_train --do_eval --evaluate_during_training
```

For '--model_name_or_path' parameter, see the paper for info about which models we use for each language. **For SEMEVAL and DURel**, the sentences in the corpora are shuffled, therefore the context is limited to sentences. For this reason **USE AN ADDITIONAL '--line_by_line' flag** when training on this corpora.

#### Extract BERT embeddings:<br/>

Extract embeddings from the preprocessed corpus in .txt for one of the corpora from the SemEval semantic change competiton:<br/>

```
python get_embeddings_scalable_semeval.py --corpus_paths pathToPreprocessedCorpusSlicesSeparatedBy';' --corpus_slices nameOfCorpusSlicesSeparatedBy';' --target_path pathToSemEvalTargetFile --language corpusLanguage --path_to_fine_tuned_model pathToFineTunedModel --embeddings_path pathToOutputEmbeddingFile
```

Extract embeddings from the preprocessed corpus in .txt for COHA, DURel or Aylien corpus:<br/>

```
python get_embeddings_scalable.py --corpus_paths pathToPreprocessedCorpusSlicesSeparatedBy';' --corpus_slices nameOfCorpusSlicesSeparatedBy';' --target_path pathToTargetFile --task chooseBetween'coha','durel','aylien' --path_to_fine_tuned_model pathToFineTunedModel --embeddings_path pathToOutputEmbeddingFile
```

This creates a pickled file containing all contextual embeddings for all target words.<br/>

#### Get results:<br/>

Conduct clustering and measure semantic shift with various methods:<br/>

```
python measure_semantic_shift.py --corpus_slices nameOfCorpusSlicesSeparatedBy';' --embeddings_path pathToInputEmbeddingFile --results_dir_path pathToOutputResultsDir --method JSD_or_WD
```

This script takes the pickled embedding file as an input and creates a csv file containing semantic change scores for each target word (from the full vocabulary or from a pre-defined list) using either Wasserstein distance or Jensen-Shannon divergence. If --get_additional_info flag is used, the script will generate additional files that are used for interpretation of the change, a file containing cluster labels for each embedding , file containing cluster centroids and a file containing context (sentence) mapped to each embedding. Note that the --get_additional_info flag can only be used if less than 100 words need to be interpreted. If your embeddings contain more than 100 words use the --define_words_to_interpret 'word1;word2;word3' flag, with which you can manually define words for which you want additional info. <br/>
To compare the output semantic change scores to the gold standard scores use the following command:<br/>

```
python evaluate.py --task chooseBetween'coha','durel','semeval' --gold_standard_path pathToGoldStandardScores --results_path pathToCSVfileWithResults --corpus_slices nameOfCorpusSlicesSeparatedBy';'
```

Extract keywords for each cluster and plot clusters distributions for interpretation:<br/>

```
python interpretation.py  --target_word targetWord --corpus_slices nameOfCorpusSlicesSeparatedBy';' --path_to_labels pathToFileWithClusterLabels --path_to_sentences pathToFileWithClusterSentences --results_dir_path pathToInterpretationResultsDir
```

This script requires a specific target word for which embeddings were generated as an input. It extracts keywords for each cluster, and plots the cluster distribution for each corpus slice.

**If something is unclear, check the default arguments for each script. If you still can't make it work, feel free to contact us :).**


