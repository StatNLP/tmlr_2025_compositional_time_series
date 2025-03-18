from random import sample
from random import seed
import numpy as np
import pickle
import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Specify Dataset')
parser.add_argument('dataset', choices=['mimic-full', 'mimic-short', 'eicu-full', 'eicu-short'])
args = parser.parse_args()
dataset = args.dataset


def syn_seq(data):
    '''
    '''
    data_look = data
    if isinstance(data, set): 
        data = sorted(data)
    else:
        data_look = set(data_look)
    
    while True:
        org_a, org_b = sample(data, 2)
        min_len = min(len(org_a), len(org_b))
        if (org_a != org_b) and (min_len > 2):
            cut = sample(range(1, min_len - 1), 1)[0]
            syn_1 = org_a[:cut] + org_b[cut:]
            syn_2 = org_b[:cut] + org_a[cut:]
            if syn_1 not in data_look: yield syn_1
            if syn_2 not in data_look: yield syn_2 


print('Reading Input: start')
path_vtse = Path(os.getcwd(), 'data_processed/' + dataset + '/time_series-vectors.pickle')
with open(path_vtse, 'rb') as f:
    train_inputs = pickle.load(f)
print('Reading Input: done')

seed_list = [41,42,43]
synthetic_size_factor = 20
N_syn_seq = synthetic_size_factor * len(train_inputs)
print('Start generating synthetic examples')
for rs in seed_list: 
    np.random.seed(rs) 
    seed(rs)
    syn_inputs = list()
    for s in syn_seq(train_inputs):
        if len(syn_inputs) % len(train_inputs) == 0: 
            print('For s' + str(rs) + ': SynDataSize is ' + str(len(syn_inputs) // len(train_inputs)) + ' of ' + str(
                   synthetic_size_factor) + '.')
        if len(syn_inputs) == N_syn_seq: break
        #trim s to model input length (48h), a symbol represents 3 consecutive vectors long
        s = s[:(48//3)]
        #si = np.array(reduce(lambda x,y: x+y, s, tuple()))
        si = np.array(s)
        si = si.reshape(si.shape[0] * si.shape[1], si.shape[2])
        if si.shape[0] < 48:
            pad_array = np.zeros((48-si.shape[0],si.shape[1]), dtype=si.dtype)
            si = np.concatenate((si, pad_array), axis=0 )
        syn_inputs.append(si)
    syn_inputs = np.stack(syn_inputs, axis=0)
    #syn_inputs should be a np.array with shape (#syn_inputs, 48, 262) for dense model
    
    path_out = Path(os.getcwd(), 'synthetic_inputs/' + dataset + '/cutmix-s' + str(rs) + '.pickle')
    with open(path_out, 'wb') as f:
        pickle.dump(syn_inputs, f)