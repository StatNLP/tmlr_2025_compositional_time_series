import numpy as np
import pickle
import os
from pathlib import Path
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='Specify Dataset')
parser.add_argument('dataset', choices=['mimic-full', 'mimic-short', 'eicu-full', 'eicu-short'])
args = parser.parse_args()
dataset = args.dataset
#dataset = 'eicu-short'

path_in = Path(os.getcwd(), 'embeddings/embd3h_' + dataset + '.pickle')
path_etse = Path(os.getcwd(), 'data_processed/' + dataset + '/time_series-embeddings.pickle')
path_vtse = Path(os.getcwd(), 'data_processed/' + dataset + '/time_series-vectors.pickle')
path_emb = Path(os.getcwd(), 'data_processed/' + dataset + '/embeddings-cluster_ready.pickle')
path_map = Path(os.getcwd(), 'data_processed/' + dataset + '/embd_to_vctrs.pickle')


def rm_padding(io, V):
    vctr = io[0]
    embd = io[1]

    for i in range(vctr.shape[0]-1, -1, -1):
        #rm vctr triples if all values are masked
        if np.sum(vctr[i,:,V:]) != 0.0: 
            i += 1
            break
    return (vctr[:i,:,:], embd[:i,:])


with open(path_in, 'rb') as f:
    dataset = pickle.load(f)

vctr_to_embd = defaultdict(set)
embd_to_vctr = defaultdict(set)
embeddings = set()
time_series_embd = list()
time_series_vctr = list()
for io in dataset:
    #io = rm_padding(io,6)
    ts_embd = list()
    ts_vctr = list()
    for i in range(io[0].shape[0]):
        vctrs = tuple(map(tuple, tuple(io[0][i])))
        embd =  tuple(io[1][i])
        #expand mapping
        vctr_to_embd[vctrs].add(embd)
        #expand inv maping
        embd_to_vctr[embd].add(vctrs)
        #expand embd set
        embeddings.add(embd)
        #collect ts emeddings
        ts_embd.append(embd)
        #collect ts vectors
        ts_vctr.append(vctrs)
    time_series_embd.append(tuple(ts_embd))
    time_series_vctr.append(tuple(ts_vctr))


#write embd |-> vctr
with open(path_map, 'wb') as f:
    pickle.dump(embd_to_vctr, f)


#write embeddings as array for clustering
embeddings = np.asarray(tuple(embeddings))
embeddings = np.ascontiguousarray(embeddings)
with open(path_emb, 'wb') as f:
    pickle.dump(embeddings, f)

#write time series data
with open(path_etse, 'wb') as f:
    pickle.dump(time_series_embd, f)

with open(path_vtse, 'wb') as f:
    pickle.dump(time_series_vctr, f)
