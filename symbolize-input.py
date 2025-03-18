'''
1. turn train inputs into set
2. generate random clusters by sample 
    
'''
from random import sample
from random import seed
from collections import defaultdict
from functools import reduce
import numpy as np
from operator import iconcat
import pickle
import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Specify Dataset')
parser.add_argument('dataset', choices=['mimic-full', 'mimic-short', 'eicu-full', 'eicu-short'])
args = parser.parse_args()
dataset = args.dataset


def create_syms(unique_blocks_arr, k, random_seed):
    '''
    '''
    seed(random_seed)
    np.random.seed(random_seed)
    #draw centroids
    centroids =  np.random.default_rng().choice(unique_blocks_arr, k)
    #fix array dims for subtraction
    centroids = np.expand_dims(centroids, axis=0)
    unique_blocks_arr = np.expand_dims(unique_blocks_arr, axis=1)
    #calculate l2-distance
    dist_mat = np.linalg.norm(unique_blocks_arr - centroids, axis=(2,3), ord='fro')
    #pick cluster with min distance
    return np.argmin(dist_mat, axis=1).tolist()


def create_maps(unique_blocks, sym_lst):
    '''
    '''
    v2s = dict(zip(unique_blocks, sym_lst))
    s2v = dict.fromkeys(sym_lst)
    
    for k, v in zip(sym_lst, unique_blocks): 
        if s2v[k] == None: s2v[k] = [] 
        s2v[k].append(v)
    return s2v, v2s


def symbolize(train_inputs: tuple, map_v2s: dict):
    '''
    '''
    return tuple(tuple(map_v2s[j] for j in i) for i in train_inputs)


print('Read input')
path_vtse = Path(os.getcwd(), 'data_processed/' + dataset + '/time_series-vectors.pickle')
with open(path_vtse, 'rb') as f:
    train_inputs = pickle.load(f)

unique_blocks = list(set(reduce(iconcat, map(list, train_inputs), [])))
unique_blocks_arr = np.asarray(unique_blocks)

seed_list = [41,42,43]
k_list = [40,80,160] 
for k in k_list:
    for s in seed_list:
        print('For K=' + str(k) + '; s=' + str(s) +': Symbolizing time series and generating s2v map')
        syms = create_syms(unique_blocks_arr, k, s)
        s2v, v2s = create_maps(unique_blocks, syms)
        sin = symbolize(train_inputs, v2s)
        
        path_sin = Path(os.getcwd(), 'symbolization/' + dataset + '/symbolized_inputs/raw-k' + str(k) +  '_s' + str(s) + '.pickle')
        path_s2v = Path(os.getcwd(), 'symbolization/' + dataset + '/s2v_maps/raw-k' + str(k) +  '_s' + str(s) + '.pickle')

        print('For K=' + str(k) + '; s=' + str(s) +': Writing symbolized time series')
        with open(path_sin, 'wb') as f:
            pickle.dump(sin, f)

        print('For K=' + str(k) + '; s=' + str(s) +': Writing s2v map')
        with open(path_s2v, 'wb') as f:
            pickle.dump(s2v, f)
