import tensorflow as tf
import torch
from tensorflow import keras
from transformers import AutoTokenizer, AutoModelWithLMHead, TFAutoModel, TFAutoModelForSequenceClassification, AutoConfig, BertForMaskedLM
from pplm_utils import *
from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm_notebook
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

from functools import partial
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import faiss
from annoy import AnnoyIndex


def make_or_load_annoy(embedding_posix, n_trees=500, index_type="angular", a=0, b=3): #a and b determine how many hidden layers of [CLS] to include in the embeddings for transformer-based embeddings
    current_dir= Path(".")
    encoded_dir = current_dir / "Encoded"
    tokenized_dir = current_dir / "Tokenized"
    embedding_name= embedding_posix.stem
    suffix= ""
    if not (current_dir / "Annoy" / f'{embedding_name}_{index_type}_{n_trees}.ann').exists() and not (current_dir / "Annoy" / f'{embedding_name}_{index_type}_{n_trees}_{a}_to_{b}.ann').exists():
        print(f"Now reading embeddings into numpy array from {str(embedding_posix)}")
        
        embeddings = np.load(str(embedding_posix))
        if embedding_name in [file.stem for file in tokenized_dir.iterdir()]:
            config = AutoConfig.from_pretrained(embedding_name)
            dim = config.dim
            print(f"Taking the Pooler (CLS) embeddings from hidden layers {a} to {b}")
            embeddings = embeddings[:, a*dim:b*dim]
            suffix = f"_{a}_to_{b}"
        d = embeddings.shape[1]
        print(f"Embeddings are {d}-dimensional")
        annoy = AnnoyIndex(d, index_type)
        
        print("Now adding vectors to index")
        for i, embedding in tqdm_notebook(enumerate(embeddings), total=embeddings.shape[0]):
            annoy.add_item(i, embedding)
        print(f"Now building index for {embedding_name}")
        annoy.build(n_trees)
        annoy.save(f'./Annoy/{embedding_name}_{index_type}_{n_trees}{suffix}.ann')
        print((f'Saved to ./Annoy/{embedding_name}_{index_type}_{n_trees}{suffix}'))
    else: #True when Index already exists
        #Read header of numpy array from disk without loading it to get its shape
        d = 512
        if embedding_name in [file.stem for file in tokenized_dir.iterdir()]: #True for huggingface models    
            suffix = f"_{a}_to_{b}"
            config = AutoConfig.from_pretrained(embedding_name)
            dim = config.dim
            d = (b-a)*dim
        elif (embedding_name.split("_")[0] == "autoencoded"):
            with open(f"./Encoded/{embedding_name}.npy", "rb") as npy:                
                version = np.lib.format.read_magic(npy)
                shape,_,_ = np.lib.format._read_array_header(npy, version)
                d = shape[1]

        print(f"Index already exists, now loading {embedding_name}_{index_type}_{n_trees}{suffix}")
        annoy = AnnoyIndex(d, index_type)
        annoy.load(f'./Annoy/{embedding_name}_{index_type}_{n_trees}{suffix}.ann')
        
    print("Now reading in argument ids")
    
    
    if embedding_name in [file.stem for file in tokenized_dir.iterdir()]: #True for huggingface models    
        print("Found tokenized pickle.")
        tokenized = pd.read_pickle(f"./Tokenized/{embedding_name}.pkl")
        arg_ids = tokenized[["id"]]
    
    elif (embedding_name.split("_")[0] == "autoencoded"):
        print("Found tokenized pickle.")
        tokenized = pd.read_pickle(f"./Tokenized/{embedding_name.split('_')[1]}.pkl")
        arg_ids = tokenized[["id"]]
    else:
        print("No tokenized pickle. Reading original dataset.")
        dataset = pd.read_pickle("./Data/dataset.pkl")
        arg_ids = dataset[["id"]]
    print("-------------------")
    print("-------ANNOY-------")
    print("-------------------")

    return annoy, arg_ids

def make_or_load_pq(embedding_posix, m=64, n_bits=8, a=0, b=3): #a and b determine how many hidden layers of [CLS] to include in the embeddings for transformer-based embeddings
    current_dir= Path(".")
    encoded_dir = current_dir / "Encoded"
    tokenized_dir = current_dir / "Tokenized"
    embedding_name= embedding_posix.stem
    suffix= ""
    if not (current_dir / "Faiss" / f'{embedding_name}_{m}_{n_bits}.faiss').exists() and not (current_dir / "Faiss" / f'{embedding_name}_{m}_{n_bits}_{a}_to_{b}.faiss').exists():
        print(f"Now reading embeddings into numpy array from {str(embedding_posix)}")
        
        embeddings = np.load(str(embedding_posix))
        d = embeddings.shape[1]
        if embedding_name in [file.stem for file in tokenized_dir.iterdir()]:
            config = AutoConfig.from_pretrained(embedding_name)
            dim = config.dim
            print(f"Taking the Pooler (CLS) embeddings from hidden layers {a} to {b}")
            embeddings = embeddings[:, a*dim:b*dim]
            embeddings = np.ascontiguousarray(embeddings)
            embeddings = embeddings.astype(np.float32)
            suffix = f"_{a}_to_{b}"
            
            
            d = embeddings.shape[1]
            print(d)
        print(d)
        print(f"Embeddings are {d}-dimensional")
        pq = faiss.IndexPQ(d, m, n_bits)
        print(embeddings.shape)
        print("Now training PQ-Index")
        pq.train(embeddings)
        
        print("Now adding embeddings to PQ-Index")
        pq.add(embeddings)
        
        faiss.write_index(pq, f'./Faiss/{embedding_name}_{m}_{n_bits}{suffix}.faiss')
        print((f'Saved to ./Faiss/{embedding_name}_{m}_{n_bits}{suffix}'))
        
    else: #True when Index already exists

        if embedding_name in [file.stem for file in tokenized_dir.iterdir()]:
            suffix = f"_{a}_to_{b}"
               
        print(f"Index already exists, now loading {embedding_name}_{m}_{n_bits}{suffix}")
        pq = faiss.read_index(f'./Faiss/{embedding_name}_{m}_{n_bits}{suffix}.faiss')
        
        
    print("Now reading in argument ids")
    if embedding_name in [file.stem for file in tokenized_dir.iterdir()]: #True for huggingface models    
        print("Found tokenized pickle.")
        tokenized = pd.read_pickle(f"./Tokenized/{embedding_name}.pkl")
        arg_ids = tokenized[["id"]]
    
    elif (embedding_name.split("_")[0] == "autoencoded"):
        print("Found tokenized pickle.")
        tokenized = pd.read_pickle(f"./Tokenized/{embedding_name.split('_')[1]}.pkl")
        arg_ids = tokenized[["id"]]
    else:
        print("No tokenized pickle. Reading original dataset.")
        dataset = pd.read_pickle("./Data/dataset.pkl")
        arg_ids = dataset[["id"]]
    print("-------------------")
    print("-------FAISS-------")
    print("-------------------")

    return pq, arg_ids

def return_args(id_set):
    l = list(id_set)
    return arguments[arguments['id'].isin(l)].copy()

def expand_mlm(model, tokenizer, query, k=10):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_STOPWORDS
    SKLEARN_STOPWORDS = set(SKLEARN_STOPWORDS)

    from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOPWORDS

    import nltk
    from ipywidgets import Output
    out = Output()
    with out:
        nltk.download('stopwords')
    from nltk.corpus import stopwords
    NLTK_STOPWORDS = set(stopwords.words('english'))

    STOP_WORDS = set.union(*[SKLEARN_STOPWORDS, SPACY_STOPWORDS, NLTK_STOPWORDS])
    from string import punctuation


    query = "What do you think? "+ query

    input_context_pro = [
    '-'+query+'\n-Yes, because of [MASK] and the benefits of [MASK] [MASK].',
    '-'+query+'\n-Absolutely, I think [MASK] is good!.',
    "-"+query+"\n-Yes, [MASK] is associated with [MASK] during [MASK]."

    ]



    input_context_con = [
    '-'+query+'\n-No, because of [MASK] and the risk of [MASK] [MASK].',
    '-'+query+'\n-Absolutely not, I think [MASK] is bad!.',
    "-"+query+"\n-No, [MASK] is associated with [MASK] during [MASK]."
    ]

    input_context_neutral = [
    query+' What about [MASK] or [MASK]?',
    query+" Don't forget about [MASK]!"
    
    ]
    hallucinations = []
    for input_context in chain(*[input_context_pro, input_context_con, input_context_neutral]):
        inp_tens = torch.tensor(tokenizer.encode(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_context)))).unsqueeze(0)
        mask_indices = np.nonzero(inp_tens.squeeze()==103).squeeze()
        preds = model(inp_tens)[0].squeeze()
        mask_indices = [mask_indices.tolist()] if type(mask_indices.tolist())!=list else mask_indices.tolist()
        top_words = []
        for i in mask_indices:
            top_words.append(torch.topk(preds[i], k=k))
        words = []
        for mask_topk in top_words:
            for token in mask_topk.indices.tolist():
                words.append(tokenizer.decode(token, clean_up_tokenization_spaces=True))
            #Interestingly, BERT was returning the ##carriage subword, obviously part of "miscarriage" in the "pregnancy" context. Further investigation needed to see how to return the full of word. Filtering out subwords for now.
            #Filter out subwords
            words = [word.replace(" ","") for word in words if not word.startswith("#")]
            words = [word for word in words if not word.endswith("#")]
            #Filter out punctuation
            words = [word for word in words if not word in punctuation]
        words = set(words)
        words = list(words.difference(STOP_WORDS))
        hallucinations.extend(words)
    hallucinations = set(hallucinations)
    hallucinations = list(hallucinations)
    return hallucinations

def expand_lm(model, tokenizer, query, print_generated=True, max_len=100, num_beams=10, num_return_sequences=3, temperature=1.6, repetition_penalty=20, top_k=100, top_p=0.4):
    query = "What do you think?"+query 
    input_context_pro = [
    "- "+query+"\n- Yes because",
    query+"The answer is yes."
    ]
    input_context_con = [
    "- "+query+"\n- No because",
    query+"The answer is no."
    ]

    input_context_neutral = [
    "- " + query + "\n- I don't know",
    "- " + query + "\n- Not sure"
    ]
    hallucinated_greedy = []

    for j, input_context in enumerate(chain(*[input_context_pro, input_context_con, input_context_neutral])):
        input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  
        L = len(input_ids[0])
        outputs =model.generate(max_length=100, input_ids=input_ids, do_sample=False, num_beams=num_beams, top_k=top_k , top_p=top_p, num_return_sequences=1, temperature=temperature, repetition_penalty=repetition_penalty)
        for i in range(1): 
            hallucinated_greedy.append(tokenizer.decode(outputs[i][L:], skip_special_tokens=True))
            if print_generated:
                print('')
                print(f'Greedily hallucinated for query {j}:\n {tokenizer.decode(outputs[i][L:], skip_special_tokens=True)}')
                print('')



    hallucinated_sampling = []

    for j, input_context in enumerate(chain(*[input_context_pro, input_context_con, input_context_neutral])):
        input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
        L = len(input_ids[0])
        outputs =model.generate(max_length=100, input_ids=input_ids, do_sample=True, num_beams=num_beams, top_k=top_k , top_p=top_p, num_return_sequences=num_return_sequences, temperature=temperature, repetition_penalty=repetition_penalty)
        for i in range(num_return_sequences):
            if print_generated:
                print(" ")
                print(f'Hallucinated {i+1} for query {j+1}: {tokenizer.decode(outputs[i][L:], skip_special_tokens=True)}')
                print(" ")
            hallucinated_sampling.append(tokenizer.decode(outputs[i][L:], skip_special_tokens=True))

    return hallucinated_greedy, hallucinated_sampling

def expand_pplm(model, tokenizer, query, print_generated=True, bag_of_words='arg_bow', length=10, stepsize=0.03, temperature=1.1,  top_k=15, num_iterations=6, num_samples=3, grad_length=1000, horizon_length=5, gm_scale=0.95, kl_scale=0.5, repetition_penalty=2, gamma=1.5, no_cuda=True, device="cpu"):
    hallucinated = []
    
    query = "What do you think?"+query 
    input_context_pro = [
    "- "+query+"\n- Yes because",
    query+"The answer is yes."
    ]
    input_context_con = [
    "- "+query+"\n- No because",
    query+"The answer is no."
    ]

    input_context_neutral = [
    "- " + query + "\n- I don't know",
    "- " + query + "\n- Not sure"
    ]
    for q in chain(*[input_context_con, input_context_pro, input_context_neutral]):
        tokenized_cond_text = tokenizer.encode("- What do you think? " + q + "\n- Yes")
        _ , pert_gen_tok_texts, _, _ = full_text_generation(model=model, tokenizer=tokenizer, context=tokenized_cond_text, bag_of_words=bag_of_words, length=length, stepsize=stepsize, temperature=temperature,  top_k=top_k, num_iterations=num_iterations, num_samples=num_samples, grad_length=grad_length, horizon_length=horizon_length, gm_scale=gm_scale, kl_scale=kl_scale, repetition_penalty=repetition_penalty, gamma=gamma, no_cuda=no_cuda, device=device)
        hallucinated.append(tokenizer.decode(pert_gen_tok_texts[0][0][len(tokenized_cond_text):]))
        if print_generated:
            print(tokenizer.decode(pert_gen_tok_texts[0][0][len(tokenized_cond_text):]))                                                                                
                                                                                          
                                                                                          
    return hallucinated
