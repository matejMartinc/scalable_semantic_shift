import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pickle
import gc
import re
from collections import defaultdict
from tokenizers import (BertWordPieceTokenizer)
import pandas as pd
import json
import nltk
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from collections import Counter



def remove_mentions(text, replace_token):
    return re.sub(r'(?:@[\w_]+)', replace_token, text)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_shifts(input_path):
    shifts_dict = {}
    df_shifts = pd.read_csv(input_path, sep=',', encoding='utf8')
    for idx, row in df_shifts.iterrows():
        shifts_dict[row['word']] = row['mean']
    return shifts_dict

def get_syntetic_shifts(input_path):
    shifts_dict = {}
    df_shifts = pd.read_csv(input_path, sep=';', encoding='utf8')
    for idx, row in df_shifts.iterrows():
        shifts_dict[row['word']] = row['JSD first-last']
    return shifts_dict

def add_embedding_to_list(previous, word_emb):
    embeds = [x[0] / x[1] for x in previous]
    cs = list(cosine_similarity(word_emb.reshape(1, -1), np.array(embeds))[0].tolist())
    if len(previous) < 200 and max(cs) < 0.99:
        max_idx = len(previous)
        previous.append((word_emb, 1))
    else:
        max_idx = cs.index(max(cs))
        old_embd, count = previous[max_idx]
        new_embd = old_embd + word_emb
        count = count + 1
        previous[max_idx] = (new_embd, count)
    return previous, max_idx


def cluster_word_embeddings_k_means(word_embeddings, k):
    clustering = KMeans(n_clusters=k, random_state=0).fit(word_embeddings)
    labels = clustering.labels_
    exemplars = clustering.cluster_centers_
    return labels, exemplars


def cluster_word_embeddings_aff_prop(word_embeddings, preference=None):
    if preference is not None:
        clustering = AffinityPropagation(preference=preference).fit(word_embeddings)
    else:
        clustering = AffinityPropagation().fit(word_embeddings)
    labels = clustering.labels_
    exemplars = clustering.cluster_centers_
    return labels, exemplars


def add_embedding_to_list_clustering(previous, word_emb, clustering):
    embeds = [x[0] / x[1] for x in previous]
    treshold = 200
    k = int(treshold/3)
    if len(previous) < treshold:
        previous.append((word_emb, 1))
    else:
        if clustering == 'aff_prop':
            _, centroids = cluster_word_embeddings_aff_prop(embeds)
        elif clustering == 'kmeans':
            cluster_word_embeddings_k_means(embeds, k)
        #print(centroids.shape)
        previous = []
        for c in centroids:
            previous.append((c, 1))
        previous.append((word_emb, 1))
    return previous, 1


def tokens_to_batches(ds, tokenizer, batch_size, max_length, target_words, lang, task):

    batches = []
    batch = []
    batch_counter = 0
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    #frequencies = defaultdict(int)
    #target_words = list(target_words.keys())

    print('Dataset: ', ds)
    counter = 0
    sent_counter = 0
    sent2count = {}
    count2sent = {}
    if task == 'syntetic':
        sent2target_sent={}

    with open(ds, 'r', encoding='utf8') as f:

        for line in f:
            counter += 1

            if counter % 500 == 0:
                print('Num articles: ', counter)

            #if counter > 50:
            #    break

            if task=='syntetic':
                text = line
            else:
                line = json.loads(line)
                text = line['content']

            #for w in target_words:
            #    if w.strip() in set(text.split()):
            #        frequencies[w] += text.count(w)

            tokenized_text = []

            #print(line)

            for sent in sent_tokenizer.tokenize(text):
                sent_counter += 1
                lsent = sent.strip().lower()
                if len(lsent.split()) > 3:
                    if task == 'syntetic':
                        target_sent = lsent
                        lsent = lsent.replace('_xx', '')
                    marked_sent = "[CLS] " + lsent + " [SEP]"
                    tokenized_sent = tokenizer.tokenize(marked_sent)
                    if len(tokenized_sent) > max_length:
                        tokenized_sent = tokenized_sent[:max_length - 1] + ['[SEP]']
                    sent = tokenizer.convert_tokens_to_string(tokenized_sent)
                    count2sent[sent_counter] = sent
                    sent2count[sent] = sent_counter
                    if task == 'syntetic':
                        sent2target_sent[sent] = target_sent
                    if len(tokenized_text) + len(tokenized_sent) > max_length:
                        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                        #print("Batch counter: ", len(tokenized_text), batch_counter, tokenized_text)
                        batch.append((indexed_tokens, tokenized_text))
                        batch_counter += 1
                        tokenized_text = tokenized_sent
                        if batch_counter % batch_size == 0:
                            batches.append(batch)
                            batch = []
                    else:
                        tokenized_text.extend(tokenized_sent)

            if len(tokenized_text) > 0:
                #print("Batch counter: ", len(tokenized_text), batch_counter, tokenized_text)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                batch.append((indexed_tokens, tokenized_text))
                batch_counter += 1
                if batch_counter % batch_size == 0:
                    batches.append(batch)
                    batch = []
            #print('-----------------------------------')
            #print()

            #uncomment this and comment the part above if you only want to use word's sentence as a context and not an entire sequence of 128 or 256 or whatever
            '''for sent in sent_tokenizer.tokenize(text):
                sent_counter += 1
                lsent = sent.strip().lower()
                if len(lsent.split()) > 3:

                    marked_sent = "[CLS] " + lsent + " [SEP]"
                    tokenized_sent = tokenizer.tokenize(marked_sent)[:max_length]

                    sent = tokenizer.convert_tokens_to_string(tokenized_sent)
                    count2sent[sent_counter] = sent
                    sent2count[sent] = sent_counter
                    batch_counter += 1
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
                    batch.append((indexed_tokens, tokenized_sent))
                    if batch_counter % batch_size == 0:
                        batches.append(batch)
                        batch = []'''

    print()
    print('Tokenization done!')
    print('len batches: ', len(batches))

    #for w, freq in frequencies.items():
    #    print(w + ': ', str(freq))

    if task == 'syntetic':
        return batches, count2sent, sent2count, sent2target_sent
    return batches, count2sent, sent2count, None


def get_token_embeddings(batches, model, batch_size):

    encoder_token_embeddings = []
    tokenized_text = []
    counter = 0

    for batch in batches:
        counter += 1
        if counter % 1000 == 0:
            print('Generating embedding for batch: ', counter)
        lens = [len(x[0]) for x in batch]
        max_len = max(lens)
        tokens_tensor = torch.zeros(batch_size, max_len, dtype=torch.long).cuda()
        segments_tensors = torch.ones(batch_size, max_len, dtype=torch.long).cuda()
        batch_idx = [x[0] for x in batch]
        batch_tokens = [x[1] for x in batch]

        for i in range(batch_size):
            length = len(batch_idx[i])
            for j in range(max_len):
                if j < length:
                    tokens_tensor[i][j] = batch_idx[i][j]

        #print("Input shape: ", tokens_tensor.shape)
        #print(tokens_tensor)

        # Predict hidden states features for each layer
        with torch.no_grad():
            model_output = model(tokens_tensor, segments_tensors)
            encoded_layers = model_output[-1][-4:]  #last four layers of the encoder

        for batch_i in range(batch_size):
            encoder_token_embeddings_example = []
            tokenized_text_example = []


            # For each token in the sentence...
            for token_i in range(len(batch_tokens[batch_i])):

                # Holds 12 layers of hidden states for each token
                hidden_layers = []

                # For each of the 12 layers...
                for layer_i in range(len(encoded_layers)):
                    # Lookup the vector for `token_i` in `layer_i`
                    vec = encoded_layers[layer_i][batch_i][token_i]

                    hidden_layers.append(vec)

                hidden_layers = torch.sum(torch.stack(hidden_layers)[-4:], 0).reshape(1, -1).detach().cpu().numpy()

                encoder_token_embeddings_example.append(hidden_layers)
                tokenized_text_example.append(batch_tokens[batch_i][token_i])

            encoder_token_embeddings.append(encoder_token_embeddings_example)
            tokenized_text.append(tokenized_text_example)


        # Sanity check the dimensions:
        #print("Number of tokens in sequence:", len(token_embeddings))
        #print("Number of layers per token:", len(token_embeddings[0]))

    return encoder_token_embeddings, tokenized_text


def get_slice_embeddings(embeddings_path, datasets, tokenizer, model, batch_size, max_length, lang, target_dict, task, scale_method):
    vocab_vectors = {}
    count2sents = {}

    for ds in datasets:

        if task == 'syntetic':
            period = ds.split('/')[-1].split('.')[0]
        else:
            period = ds.split('_')[1].split('.')[0]

        all_batches,  count2sent, sent2count, sent2target_sense = tokens_to_batches(ds, tokenizer, batch_size, max_length, target_dict, lang, task)

        count2sents[period] = count2sent
        targets = set(list(target_dict.keys()))
        chunked_batches = chunks(all_batches, 1000)
        num_chunk = 0

        for batches in chunked_batches:
            num_chunk += 1
            print('Chunk ', num_chunk)

            #get list of embeddings and list of bpe tokens
            encoder_token_embeddings, tokenized_text = get_token_embeddings(batches, model, batch_size)

            splitted_tokens = []
            encoder_splitted_array = np.zeros((1, 768))
            prev_token = ""
            encoder_prev_array = np.zeros((1, 768))
            sent_tokens = []

            #go through text token by token
            for example_idx, example in enumerate(tokenized_text):

                #be careful, this only work cause one example for syntetic is only one sentence and not the whole sequence!
                if task=="syntetic":
                    start = example.index("[CLS]")
                    finish = example.index("[SEP]")
                    syn_sent = tokenizer.convert_tokens_to_string(example[start:finish + 1])
                    target_sent = sent2target_sense[syn_sent]
                for i, token_i in enumerate(example):
                    if token_i == "[CLS]":
                        last_start = i
                    elif token_i =="[SEP]":
                        last_finish = i
                        #print('Example: ', len(example), example)
                        sentence = tokenizer.convert_tokens_to_string(example[last_start:last_finish + 1])
                        #we ignore sents that span across two sequences
                        if sentence.startswith('[CLS]') and sentence.endswith('[SEP]'):
                            #print('Sentence: ', sentence)
                            sentence = sent2count[sentence]
                            #print('Count: ', sentence)
                            for sent_token, sent_idx in sent_tokens:
                                #print(sent_token, count2sent[sentence])
                                if sent_idx in vocab_vectors[sent_token][period + '_text']:
                                    vocab_vectors[sent_token][period + '_text'][sent_idx].append(sentence)
                                else:
                                    vocab_vectors[sent_token][period + '_text'][sent_idx] = [sentence]
                            sent_tokens = []

                    encoder_array = encoder_token_embeddings[example_idx][i]

                    #word is split into parts
                    if token_i.startswith('##') or token_i == '-' or (len(splitted_tokens) > 0 and splitted_tokens[-1] == '-'):

                        #add words prefix (not starting with ##) to the list
                        if prev_token:
                            splitted_tokens.append(prev_token)
                            prev_token = ""
                            encoder_splitted_array = encoder_prev_array



                        #add word to splitted tokens array and add its embedding to splitted_array
                        splitted_tokens.append(token_i)
                        encoder_splitted_array += encoder_array


                    #word is not split into parts
                    else:
                        if token_i in targets:
                            if task=='syntetic':
                                if token_i + '_xx' not in target_sent:
                                    #print("Token i", token_i, target_sent)
                                    continue
                            #print(token_i)
                            if i == len(example) - 1 or not example[i + 1].startswith('##'):
                                if token_i in vocab_vectors:
                                    #print("In vocab: ", token_i + '_' + period, list(vocab_vectors.keys()))
                                    if period in vocab_vectors[token_i]:
                                        previous = vocab_vectors[token_i][period]
                                        if scale_method=='average':
                                            new, new_idx = add_embedding_to_list(previous, encoder_array.squeeze())
                                        else:
                                            new, new_idx = add_embedding_to_list_clustering(previous, encoder_array.squeeze(), scale_method)
                                        vocab_vectors[token_i][period] = new
                                        sent_tokens.append((token_i, new_idx))
                                    else:
                                        vocab_vectors[token_i][period] = [(encoder_array.squeeze(), 1)]
                                        vocab_vectors[token_i][period + '_text'] = {}
                                        sent_tokens.append((token_i, 0))
                                else:
                                    #print("Not in vocab yet: ", token_i + '_' + period, list(vocab_vectors.keys()))
                                    vocab_vectors[token_i] = {period:[(encoder_array.squeeze(), 1)], period + '_text': {}}
                                    sent_tokens.append((token_i, 0))
                        #check if there are words in splitted tokens array, calculate average embedding and add the word to the vocabulary
                        if splitted_tokens:

                            encoder_sarray = encoder_splitted_array / len(splitted_tokens)
                            stoken_i = "".join(splitted_tokens).replace('##', '')

                            if stoken_i in targets:
                                if task == 'syntetic':
                                    if stoken_i + '_xx' not in target_sent:
                                        #print("Stoken i", stoken_i, target_sent)
                                        continue
                                if stoken_i in vocab_vectors:
                                    #print("S In vocab: ", stoken_i + '_' + period, list(vocab_vectors.keys()))
                                    if period in vocab_vectors[stoken_i]:
                                        previous = vocab_vectors[stoken_i][period]
                                        if scale_method == 'average':
                                            new, new_idx = add_embedding_to_list(previous, encoder_sarray.squeeze())
                                        else:
                                            new, new_idx = add_embedding_to_list_clustering(previous, encoder_array.squeeze(), scale_method)
                                        vocab_vectors[stoken_i][period] = new
                                        sent_tokens.append((stoken_i, new_idx))
                                    else:
                                        vocab_vectors[stoken_i][period] = [(encoder_sarray.squeeze(), 1)]
                                        vocab_vectors[stoken_i][period + '_text'] = {}
                                        sent_tokens.append((stoken_i, 0))
                                else:
                                    #print("S Not in vocab yet: ", stoken_i + '_' + period, list(vocab_vectors.keys()))
                                    vocab_vectors[stoken_i] = {period: [(encoder_sarray.squeeze(), 1)], period + '_text': {}}
                                    sent_tokens.append((stoken_i, 0))

                            splitted_tokens = []
                            encoder_splitted_array = np.zeros((1, 768))

                        encoder_prev_array = encoder_array
                        prev_token = token_i

            del encoder_token_embeddings
            del tokenized_text
            del batches
            gc.collect()

            '''for k, v in vocab_vectors.items():
                print(k)
                input = v[0]
                encoder = v[1]
                context = v[2]
                print(len(input))
                print(len(encoder))
                print(len(context))
                print(context[0])'''

        print('Sentence embeddings generated.')

    print("Length of vocab after training: ", len(vocab_vectors.items()))

    with open(embeddings_path.split('.')[0] + '.pickle', 'wb') as handle:
        pickle.dump([vocab_vectors, count2sents], handle, protocol=pickle.HIGHEST_PROTOCOL)

    gc.collect()


if __name__ == '__main__':
    task = 'coha'
    lang = 'English'
    #scale method can be 'average', 'aff_prop' or 'kmeans'
    scale_method = 'aff_prop'
    batch_size = 16
    max_length = 256

    if task=='coha':
        datasets = ['data/coha/coha_1960.txt',
                    'data/coha/coha_1990.txt', ]
        state_dict = torch.load("models/model_coha_epoch_5/checkpoint-69350/pytorch_model.bin")
        embeddings_path = 'embeddings/coha_5_yearly_fine_tuned_aff_prop.pickle'
        shifts_dict = get_shifts('data/coha/Gulordava_word_meaning_change_evaluation_dataset.csv')
    elif task=='aylien':
        '''datasets = ['data/aylien/aylien_january_balanced.txt',
                    'data/aylien/aylien_february_balanced.txt',
                    'data/aylien/aylien_march_balanced.txt',
                    'data/aylien/aylien_april_balanced.txt',]'''
        datasets = ['data/aylien/aylien_cnn.txt',
                    'data/aylien/aylien_fox.txt']
        state_dict = torch.load("models/model_aylien/checkpoint-173935/pytorch_model.bin")
        embeddings_path = 'embeddings/aylien_monthly_balanced_fine_tuned.pickle'
        shifts_dict = get_shifts('data/aylien/vocab.csv')

    elif task=='syntetic':
        datasets = ['data/syntetic_data/raw/00.txt',
                    'data/syntetic_data/raw/01.txt',
                    'data/syntetic_data/raw/02.txt',
                    'data/syntetic_data/raw/03.txt',
                    'data/syntetic_data/raw/04.txt',
                    'data/syntetic_data/raw/05.txt',
                    'data/syntetic_data/raw/06.txt',
                    'data/syntetic_data/raw/07.txt',
                    'data/syntetic_data/raw/08.txt',
                    'data/syntetic_data/raw/09.txt',]

        embeddings_path = 'embeddings/syntetic_pretrained.pickle'
        shifts_dict = get_syntetic_shifts('data/syntetic_data/syntetic_gs.csv')


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    if task=='syntetic':
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    else:
        model = BertModel.from_pretrained('bert-base-uncased', state_dict=state_dict, output_hidden_states=True)

    model.cuda()
    model.eval()

    #print(shifts_dict.items())
    get_slice_embeddings(embeddings_path, datasets, tokenizer, model, batch_size, max_length, lang, shifts_dict, task, scale_method)





