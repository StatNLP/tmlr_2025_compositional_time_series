import os
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import argparse

parser = argparse.ArgumentParser(description='Specify Dataset')
parser.add_argument('dataset', choices=['mimic-full', 'mimic-short', 'eicu-full', 'eicu-short'])
args = parser.parse_args()
dataset = args.dataset

path_emb = Path(os.getcwd(), 'data_processed/' + dataset + '/embeddings-cluster_ready.pickle')
path_clstr_selected = Path(os.getcwd(), 'symbolization/' + dataset + '/embd-selected_k.pickle')

with open(path_emb, 'rb') as f:
    embeddings = pickle.load(f)


#1 Studentize data
scaler = StandardScaler()
scaled_embds = scaler.fit_transform(embeddings)

#2 Fit k-means
kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42}

k_list = [10,20,40,80,160] 
clusterings = dict()
for k in k_list:
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    clusterings[k] = kmeans.fit(scaled_embds)

#4 embd |-> clstr
path_tse = Path(os.getcwd(), 'data_processed/' + dataset + '/time_series-embeddings.pickle')
with open(path_tse, 'rb') as f:
    train_inputs = pickle.load(f)

path_e2v = Path(os.getcwd(), 'data_processed/' + dataset + '/embd_to_vctrs.pickle')
with open(path_e2v, 'rb') as f:
    map_e2v = pickle.load(f)



for k in k_list:
    k_symbols = tuple(map(int, clusterings[k].labels_))
    map_e2s = dict(zip(tuple(map(tuple, embeddings)), k_symbols))
    #1. symbolize time series
    print('For K=' + str(k) + '; s=42: Symbolizing time series')
    sin = list()
    for ts in train_inputs:
        sym_seq = tuple(map_e2s[embd] for embd in ts)
        sin.append(sym_seq)
    #2. generate sym |-> vec
    print('For K=' + str(k) + '; s=42: Generating s2v map')
    s2v = defaultdict(list)
    for embd,vec in map_e2v.items():
        sym = map_e2s[embd] 
        s2v[sym].extend(vec)
       
    path_sin = Path(os.getcwd(), 'symbolization/' + dataset + '/symbolized_inputs/embedding3h-k' + str(k) + '_s42.pickle')
    path_s2v = Path(os.getcwd(), 'symbolization/' + dataset + '/s2v_maps/embedding3h-k' + str(k) + '_s42.pickle')

    print('For K=' + str(k) + '; s=42: Writing symbolized time series')
    with open(path_sin, 'wb') as f:
        pickle.dump(tuple(sin), f)

    print('For K=' + str(k) + '; s=42: Writing s2v map')
    with open(path_s2v, 'wb') as f:
        pickle.dump(s2v, f)