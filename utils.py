import tensorflow as tf
from tensorflow import keras
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoConfig
from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm_notebook
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def make_or_load_annoy(embedding_posix, n_trees=500, index_type="angular", a=0, b=3): #a and b determine how many hidden layers of [CLS] to include in the embeddings for transformer-based embeddings
    current_dir= Path(".")
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
        print("Found tokenized pickle.1")
        tokenized = pd.read_pickle(f"./Tokenized/{embedding_name}.pkl")
        arg_ids = tokenized[["id"]]
    
    elif (embedding_name.split("_")[0] == "autoencoded"):
        print("Found tokenized pickle.2")
        tokenized = pd.read_pickle(f"./Tokenized/{embedding_name.split('_')[1]}.pkl")
        arg_ids = tokenized[["id"]]
    else:
        print("No tokenized pickle. Reading original dataset.")
        dataset = pd.read_pickle("./Data/dataset.pkl")
        arg_ids = dataset[["id"]]
    print("-------------------")
    print("-------------------")
    print("-------------------")
    print("-------------------")

    return annoy, arg_ids

def make_or_load_pq(embedding_posix, m=64, n_bits=8, a=0, b=3): #a and b determine how many hidden layers of [CLS] to include in the embeddings for transformer-based embeddings
    current_dir= Path(".")
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

    print("-------------------")

    return pq, arg_ids