import os
import pickle
from pathlib import Path
from collections import defaultdict
import cds_algorithm as cds
from random import choice
from itertools import chain
from functools import reduce
import numpy as np
from re import findall
from random import seed
import argparse

parser = argparse.ArgumentParser(description='Specify Dataset')
parser.add_argument('dataset', choices=['mimic-full', 'mimic-short', 'eicu-full', 'eicu-short'])
args = parser.parse_args()
dataset = args.dataset

#cds parameters:
window_size = 1 
max_frg_len = 4
max_gaps = 1

synthetic_size_factor = 20

symbolizations = os.listdir(Path(os.getcwd(), 'symbolization/' + dataset +'/s2v_maps'))

print('Read Gold train inputs')
path_vtse = Path(os.getcwd(), 'data_processed/' + dataset + '/time_series-vectors.pickle')
with open(path_vtse, 'rb') as f:
    train_inputs = pickle.load(f)

train_inputs = np.array(train_inputs)

for symbo in symbolizations:
    #set seeds
    rs = int(findall('s(41|42|43)[.]',symbo)[0])
    np.random.seed(rs)
    seed(rs)
    #read inputs
    print('For ' + symbo + ': Read input: start')
    path_sin = Path(os.getcwd(), 'symbolization/' + dataset + '/symbolized_inputs/' + symbo)
    with open(path_sin, 'rb') as f:
        sin = pickle.load(f)
    print('For ' + symbo + ': Read input: done')
    #generate symbolization to index map
    #Remark: the ordering of sin and train set must be identical!!  
    ss2gsi = defaultdict(list)
    for i in range(len(sin)):
        ss2gsi[sin[i]].append(i)
    #build caches
    print('For ' + symbo + ': Build caches')
    f2t, t2f, e2t = cds.build_caches(sin, window_size, cds.total_fragments, max_frg_len=max_frg_len, rm_solitaire=False)
    #f2t, t2f, e2t = cds.build_caches(sin, window_size, cds.batch_fragments, min_length=2, max_length=max_frg_len, max_gaps=max_gaps)
    print('Building caches done!')

    N_syn_seq = synthetic_size_factor * len(sin)
    syn_inputs = list()
    for s in cds.syn_seq(f2t, t2f, e2t, ss2gsi, train_inputs, window_size):
        if len(syn_inputs) % len(sin) == 0: 
            print('For ' + symbo + ': SynDataSize is ' + str(len(syn_inputs) // len(sin)) + ' of ' + str(
                  synthetic_size_factor) + '.')
        if len(syn_inputs) == N_syn_seq: break
        #trim s to model input length (48h), a symbol represents 3 consecutive vectors long
        if s.shape[0] < 48:
            pad_array = np.zeros((48-s.shape[0],s.shape[1]), dtype=s.dtype)
            s = np.concatenate((s, pad_array), axis=0)
        else:
            s = s[:48,:]
        syn_inputs.append(s)
    syn_inputs = np.stack(syn_inputs, axis=0)

    print('For ' + symbo + ': Write synthetic data')
    symbo_out = symbo.replace('.', '_w' + str(window_size) + '.')
    symbo_out = symbo_out.replace('.', '_f' + str(max_frg_len) + '.')    
    path_out = Path(os.getcwd(), 'synthetic_inputs/' + dataset + '/cds_' + symbo_out)
    with open(path_out, 'wb') as f:
        pickle.dump(syn_inputs, f)



    
