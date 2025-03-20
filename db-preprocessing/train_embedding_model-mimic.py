#!/usr/bin/env python
# coding: utf-8

# *ToDo*
# 
# 02) pred_window und obs_window ausprobieren.
# 03) output nochmal genau anschauen.
# 04) was ist maskiert anschauen - und wie genau?
# 05) VRAM wenig ausgelastet. Batch size mal mit 320 ausprobieren, aber auch durchziehen.
# 06) Task pred: scheduler einfügen.
# 07) Analyse each var (sparsity)
# 08) loss per var/quality of varis loss (is this already ablation to forecast only one var)
# 09) Include sepsis definition
# 10) 

# In[1]:


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
tqdm.pandas()
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau


# In[2]:


# data settings
test_cond = 0

if test_cond == 1:
    data_path = 'mimic_iii_preprocessed_new_z_err20092.pkl'
    sample_divisor = 100
    number_of_epochs = 20
else:
    data_path = 'mimic_iii_preprocessed_new_z_err20092.pkl'
    sample_divisor = 1
    number_of_epochs = 1000


# ## Load forecast dataset into matrices.

# In[3]:


def inv_list(l, start=0):  # Create vind
    d = {}
    for i in range(len(l)):
        d[l[i]] = i+start
    return d


def f(x):
    mask   = [0 for i in range(V)]
    values = [0 for i in range(V)]
    #print('hi', mask , V)
    #mask, values = np.zeros(V), np.zeros(V)
    #print('hi', mask)
    for vv in x:  # tuple of ['vind','value']
        v = int(vv[0])-1  # shift index of vind
        mask[v] = 1
        values[v] = vv[1]  # get value
    return values+mask  # concat


def pad(x):
    if len(x) > 880:
        print(len(x))
    return x+[0]*(fore_max_len-len(x))


# In[4]:


pred_window_old = 12  # hours
obs_windows = range(0, 1, 1)

# Read data.
data, oc, train_ind, valid_ind, test_ind = pickle.load(open(data_path, 'rb'))
# Remove test patients.
data = data.merge(oc[['ts_ind', 'SUBJECT_ID']], on='ts_ind', how='left')
test_sub = oc.loc[oc.ts_ind.isin(test_ind)].SUBJECT_ID.unique()
data = data.loc[~data.SUBJECT_ID.isin(test_sub)]
oc = oc.loc[~oc.SUBJECT_ID.isin(test_sub)]
data.drop(columns=['SUBJECT_ID', 'TABLE'], inplace=True)
# Fix age.
data.loc[(data.variable == 'Age') & (data.value > 200), 'value'] = 91.4
# Get static data with mean fill and missingness indicator.
static_varis = ['Age', 'Gender']
ii = data.variable.isin(static_varis)
static_data = data.loc[ii]  # static data are 'Age' and 'Gender'
data = data.loc[~ii]  # ~ binary flip
# print('data\n',data)

static_var_to_ind = inv_list(static_varis)  # {'Age': 0, 'Gender': 1}
D = len(static_varis)  # 2 by definition
N = data.ts_ind.max()+1  # /= 12: 52861
demo = np.zeros((int(N), int(D)))
for row in tqdm(static_data.itertuples()):
    demo[int(row.ts_ind), static_var_to_ind[row.variable]] = row.value
# print('Demo after tqdm command \n',demo[:10])
# Normalize static data.
means = demo.mean(axis=0, keepdims=True)  # quite sparse
stds = demo.std(axis=0, keepdims=True)
stds = (stds == 0)*1 + (stds != 0)*stds
demo = (demo-means)/stds
# print('Demo after normalisation \n',demo[:10])
# Get variable indices.
varis = sorted(list(set(data.variable)))
V = len(varis)  # 129 for \=12 with varis all variables except for static ones
# print('varis', varis, V)
var_to_ind = inv_list(varis, start=1)
data['vind'] = data.variable.map(var_to_ind)
data = data[['ts_ind', 'vind', 'hour', 'value']].sort_values(by=['ts_ind', 'vind', 'hour'])
# print('data vind\n', data['vind'], '\n data\n',data)
# Find max_len.
fore_max_len = 880  # hard coded max_len of vars
# Get forecast inputs and outputs.
fore_times_ip = []
fore_values_ip = []
fore_varis_ip = []
fore_op = []
fore_op_awesome = []
fore_inds = []
for w in tqdm(range(3, 45, 3)):  # range(20, 124, 4), pred_window=2
    pred_data = data.loc[(data.hour>=w)&(data.hour<=w+3)]
    pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value':'first'}).reset_index()
    pred_data['vind_value'] = pred_data[['vind', 'value']].values.tolist()
    pred_data = pred_data.groupby('ts_ind').agg({'vind_value':list}).reset_index()
    pred_data['vind_value'] = pred_data['vind_value'].apply(f)   

    obs_data = data.loc[(data.hour < w) & (data.hour >= w-3)]
    obs_data = obs_data.loc[obs_data.ts_ind.isin(pred_data.ts_ind)]
    obs_data = obs_data.groupby('ts_ind').head(fore_max_len)
    obs_data = obs_data.groupby('ts_ind').agg({'vind': list, 'hour': list, 'value': list}).reset_index()
    #Michi: Hier hole ich mir die 24 Stunden vor, und 24 Stunden nach einem bestimmten Zeitpunkt raus
    for pred_window  in range(-3, 3, 1):
        pred_data = data.loc[(data.hour >= w+pred_window) & (data.hour <= w+1 +pred_window)]
        pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value': 'first'}).reset_index()
        pred_data['vind_value'+str(pred_window)] = pred_data[['vind', 'value']].values.tolist()
        pred_data = pred_data.groupby('ts_ind').agg({'vind_value'+str(pred_window): list}).reset_index()
        pred_data['vind_value'+str(pred_window)] = pred_data['vind_value'+str(pred_window)].apply(f)  # 721 entries with 2*129 vind_values
        obs_data = obs_data.merge(pred_data, on='ts_ind')

    for col in ['vind', 'hour', 'value']:
        obs_data[col] = obs_data[col].apply(pad)
    fore_op_awesome.append(np.array(list([list(obs_data['vind_value'+str(pred_window)]) for pred_window in range(-3, 3, 1)])))
    #fore_op.append(np.array(list(obs_data.vind_value)))
    fore_inds.append(np.array([int(x) for x in list(obs_data.ts_ind)]))
    fore_times_ip.append(np.array(list(obs_data.hour)))
    fore_values_ip.append(np.array(list(obs_data.value)))
    fore_varis_ip.append(np.array(list(obs_data.vind)))
    
del data
fore_times_ip = np.concatenate(fore_times_ip, axis=0)
fore_values_ip = np.concatenate(fore_values_ip, axis=0)
fore_varis_ip = np.concatenate(fore_varis_ip, axis=0)

fore_op_awesome = np.concatenate(fore_op_awesome, axis=1)
fore_op = np.swapaxes(fore_op_awesome, 0, 1)
print(fore_op.shape)

#raise Exception
fore_inds = np.concatenate(fore_inds, axis=0)
fore_demo = demo[fore_inds]
# Get train and valid ts_ind for forecast task.
train_sub = oc.loc[oc.ts_ind.isin(train_ind)].SUBJECT_ID.unique()
valid_sub = oc.loc[oc.ts_ind.isin(valid_ind)].SUBJECT_ID.unique()
rem_sub = oc.loc[~oc.SUBJECT_ID.isin(np.concatenate((train_ind, valid_ind)))].SUBJECT_ID.unique()
bp = int(0.8*len(rem_sub))
train_sub = np.concatenate((train_sub, rem_sub[:bp]))
valid_sub = np.concatenate((valid_sub, rem_sub[bp:]))
train_ind = oc.loc[oc.SUBJECT_ID.isin(train_sub)].ts_ind.unique()  # Add remaining ts_ind s of train subjects.
valid_ind = oc.loc[oc.SUBJECT_ID.isin(valid_sub)].ts_ind.unique()  # Add remaining ts_ind s of train subjects.
# Generate 3 sets of inputs and outputs.
train_ind = np.argwhere(np.in1d(fore_inds, train_ind)).flatten()
valid_ind = np.argwhere(np.in1d(fore_inds, valid_ind)).flatten()
fore_train_ip = [ip[train_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip]]
fore_valid_ip = [ip[valid_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip]]
del fore_times_ip, fore_values_ip, fore_varis_ip, demo, fore_demo
fore_train_op = fore_op[train_ind]
fore_valid_op = fore_op[valid_ind]
del fore_op

print('lengths of rem_sub, fore_train_ip[1], fore_valid_ip[0]')
print(len(rem_sub), fore_train_ip[1].shape, fore_valid_ip[0].shape)


# In[ ]:



fore_max_len = 880
# Read data.
# data_path = '/home/mitarb/fracarolli/files/230613_STraTS_preprocessed/mimic_iii_preprocessed.pkl'
data, oc, train_ind, valid_ind, test_ind = pickle.load(open(data_path, 'rb'))
means_stds = data.groupby("variable").agg({"mean":"first", "std":"first"})
#print(means_stds)
mean_std_dict = dict()
print(means_stds.keys)
for pos, row in means_stds.iterrows():
    print(pos)
    print(row)
    mean_std_dict[pos] = (float(row["mean"]), float(row["std"]))
print(mean_std_dict)
#raise Exception
# Filter labeled data in first 24h.
data = data.loc[data.ts_ind.isin(np.concatenate((train_ind, valid_ind, test_ind), axis=-1))]
data = data.loc[(data.hour >= 0) & (data.hour <= 48)]
old_oc = oc
oc = oc.loc[oc.ts_ind.isin(np.concatenate((train_ind, valid_ind, test_ind), axis=-1))]
# Fix age.
data.loc[(data.variable == 'Age') & (data.value > 200), 'value'] = 91.4
# Get y and N.
y = np.array(oc.sort_values(by='ts_ind')['in_hospital_mortality']).astype('float32')
print(y.shape)
#raise Exception
data['ts_ind'] = data['ts_ind'].astype(int) 
N = int(data.ts_ind.max() + 1)

# Create demographic/static data
# Get static data with mean fill and missingness indicator.
static_varis = ['Age', 'Gender']
ii = data.variable.isin(static_varis)
static_data = data.loc[ii]
data = data.loc[~ii]  # reduce data to non-demo/non-static
static_var_to_ind = inv_list(static_varis)
D = len(static_varis)
demo = np.zeros((int(N), D))
for row in tqdm(static_data.itertuples()):
    demo[int(row.ts_ind), static_var_to_ind[row.variable]] = row.value
# Normalize static data.
means = demo.mean(axis=0, keepdims=True)
stds = demo.std(axis=0, keepdims=True)
stds = (stds == 0)*1 + (stds != 0)*stds
demo = (demo-means)/stds

# Trim to max len.
data = data.sample(frac=1)
data = data.groupby('ts_ind').head(880)

# Get N, V, var_to_ind.
#N = data.ts_ind.max() + 1
varis = sorted(list(set(data.variable)))
# V = len(varis)
var_to_ind = inv_list(varis, start=1)
data['vind'] = data.variable.map(var_to_ind)
data = data[['ts_ind', 'vind', 'hour', 'value']].sort_values(by=['ts_ind', 'vind', 'hour'])
# Add obs index.
data = data.sort_values(by=['ts_ind']).reset_index(drop=True)
data = data.reset_index().rename(columns={'index': 'obs_ind'})
data = data.merge(data.groupby('ts_ind').agg({'obs_ind': 'min'}).reset_index().rename(columns={
                                                            'obs_ind': 'first_obs_ind'}), on='ts_ind')
data['obs_ind'] = data['obs_ind'] - data['first_obs_ind']
# Find max_len.
max_len = data.obs_ind.max()+1
print('max_len', max_len)
# Generate times_ip and values_ip matrices.
times_inp = np.zeros((N, max_len), dtype='float32')
values_inp = np.zeros((N, max_len), dtype='float32')
varis_inp = np.zeros((N, max_len), dtype='int32')
for row in tqdm(data.itertuples()):
    ts_ind = row.ts_ind
    l = row.obs_ind
    times_inp[ts_ind, l] = row.hour
    values_inp[ts_ind, l] = row.value
    varis_inp[ts_ind, l] = row.vind

w=0

fore_in = []

pred_data = data.loc[(data.hour>=0)&(data.hour<=48)]
pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value':'first'}).reset_index()
pred_data['vind_value'] = pred_data[['vind', 'value']].values.tolist()
pred_data = pred_data.groupby('ts_ind').agg({'vind_value':list}).reset_index()
pred_data['vind_value'] = pred_data['vind_value'].apply(f)   

obs_data = data.loc[(data.hour < 48) & (data.hour >= 0)]
resultdict = dict()
for ts_ind in obs_data.ts_ind:
    resultdict[ts_ind] = [[]]
obs_data = obs_data.loc[obs_data.ts_ind.isin(pred_data.ts_ind)]
obs_data = obs_data.groupby('ts_ind').head(fore_max_len)
obs_data = obs_data.groupby('ts_ind').agg({'vind': list, 'hour': list, 'value': list}).reset_index()

for pred_window  in range(0, 48, 1):
    print(pred_window)
    #print(w+1 +pred_window)
    #print(w+pred_window)
    pred_data = data.loc[(data.hour >= w+pred_window) & (data.hour <= w+1 +pred_window)]
    pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value': 'first'}).reset_index()
    pred_data['vind_value'+str(pred_window)] = pred_data[['vind', 'value']].values.tolist()
    pred_data = pred_data.groupby('ts_ind').agg({'vind_value'+str(pred_window): list}).reset_index()
    pred_data['vind_value'+str(pred_window)] = pred_data['vind_value'+str(pred_window)].apply(f)  # 721 entries with 2*129 vind_values
    obs_data = obs_data.merge(pred_data, on='ts_ind')
    #print(list(data['vind_value'+str(pred_window)]))
    #print(len(list(data['vind_value'+str(pred_window)])))
    #print(len(list(data['vind_value'+str(pred_window)])[0]))

blub = (np.array(list([list(obs_data['vind_value'+str(pred_window)]) for pred_window in range(0, 48, 1)])))
print("bleb", blub.shape)
op = np.swapaxes(blub, 0, 1)

weird_oc = oc.loc[oc.ts_ind.isin(obs_data.ts_ind)]
y = np.array(weird_oc.sort_values(by='ts_ind')['in_hospital_mortality']).astype('float32')
print(y.shape)
data.drop(columns=['obs_ind', 'first_obs_ind'], inplace=True)

train_ind = [x for x in train_ind if x < op.shape[0]]
valid_ind = [x for x in valid_ind if x < op.shape[0]]
test_ind = [x for x in test_ind if x < op.shape[0]]

train_input = op[train_ind]
valid_input = op[valid_ind]
test_input = op[test_ind]

# Generate 3 sets of inputs and outputs.
train_ip = [ip[train_ind] for ip in [demo, times_inp, values_inp, varis_inp]]
valid_ip = [ip[valid_ind] for ip in [demo, times_inp, values_inp, varis_inp]]
test_ip = [ip[test_ind] for ip in [demo, times_inp, values_inp, varis_inp]]
del demo, times_inp, values_inp, varis_inp  # warum wird demo nicht gelöscht?

if test_cond == 1:
    tr_ind = [divmod(tr, 12)[0] for tr in train_ind]
    va_ind = [divmod(tr, 12)[0] for tr in valid_ind]
    te_ind = [divmod(tr, 12)[0] for tr in test_ind]
    
    train_op = y[tr_ind]  # is a problem for the test case...
    valid_op = y[va_ind]
    test_op = y[te_ind]
else:
    train_op = y[train_ind]  # is a problem for the test case...
    valid_op = y[valid_ind]
    test_op = y[test_ind]
print('y is:', y)
del y
 


# In[ ]:


# In[ ]:


factor_indices1 = [var_to_ind[x]-1 for x in ["HR", "RR", "SBP", "DBP", "MBP", "O2 Saturation"]]
factor_indices2 = factor_indices1 + [V+x for x in factor_indices1]


# In[ ]:


#Michi: die ganzen Argparses könnte man noch anders implementieren, wollte es nur schnell zum laufen bringen.
import argparse
parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model', type=str, required=False, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=False, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=3, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=3, help='prediction sequence length')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers 
#Michi: bisher nur default=3 zum laufen gebracht
parser.add_argument('--embed_type', type=int, default=3, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=262, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=131, help='decoder input size')
parser.add_argument('--c_out', type=int, default=131, help='output size')
parser.add_argument('--d_model', type=int, default=50, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=False)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args(args=[])
import importlib


import matplotlib.pyplot as plt


# In[ ]:


import models.InformerAutoregressiveFull as autoformer
importlib.reload(autoformer)
model = autoformer.Model(args).cuda()

lr, batch_size, samples_per_epoch, patience = 0.0005, 32, int(102400/sample_divisor), 6
d, N, he, dropout = 50, 2, 4, 0.2
V=131
print('number of parameters: ', V)


#a = summary(model, [(32, 2), (32, 880), (32, 880), (32, 880)],  # shape of fore_train_ip
#            dtypes=[torch.float, torch.float, torch.float, torch.long])
#print(a)  # Model summary
# raise Exception
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
# Pretrain fore_model.
best_val_loss = np.inf
N_fore = len(fore_train_op)
#print(N_fore)
fore_savepath = 'informer_mimic_full_3hours.pytorch'
loss_func = torch.nn.MSELoss(reduction="none")

# torch.compile(model)
for e in range(number_of_epochs):
    e_indices = np.random.choice(range(N_fore), size=samples_per_epoch, replace=False)
    e_loss = 0
    pbar = tqdm(range(0, len(e_indices), batch_size))
    model.train()
    for start in pbar:
        ind = e_indices[start:start+batch_size]

        matrix = torch.tensor(fore_train_op[ind], dtype=torch.float32).cuda()
        #torch.Size([32, 48, 258])
        input_matrix = matrix[:, :3, :262]
        #torch.Size([32, 24, 129])
        input_mark = torch.arange(0, input_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        input_mask = matrix[:, :3, 131:]
        #torch.Size([32, 24, 129])
        output_matrix = matrix[:, 3:, :131]
        #torch.Size([32, 24, 129])
        
        output_mark = torch.arange(0, output_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        output_mask = matrix[:, 3:, 131:]
        #torch.Size([32, 24, 129])
        dec_inp = torch.zeros_like(output_matrix[:, -args.pred_len:, :]).float()
        #print(dec_inp.size())
        #dec_inp = torch.cat([output_matrix[:, :args.label_len, :], dec_inp], dim=1).float().cuda()
        #raise Exception
        #print(str(type(model)))
        #print(model)
        if 'Linear' not in str(type(model)) and "Autoregressive" in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark, tgt=output_matrix, trainn=False, backprop=True)#, enc_self_mask=input_mask, dec_self_mask=output_mask)
        elif "Linear" not in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark)#, enc_self_mask=input_mask, dec_self_mask=output_mask)

        else:
            output = model(input_matrix)[:, :, :131]
        loss = output_mask[:, -args.pred_len:, :]*(
        output-output_matrix[:, -args.pred_len:, :])**2
        loss = loss.sum(axis=-1).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        e_loss += loss.detach()
        pbar.set_description('%f' % e_loss)
    val_loss = 0
    #raise Exception
    loss_list = []
    model.eval()
    pbar = tqdm(range(0, len(fore_valid_op), 32))  # len(fore_valid_op)           ####################   maybe also batch_size instead of 32
    for start in pbar:
        matrix = torch.tensor(fore_valid_op[start:start+batch_size], dtype=torch.float32).cuda()
        #torch.Size([32, 48, 258])
        input_matrix = matrix[:, :3, :262]
        #torch.Size([32, 24, 129])
        input_mark = torch.arange(0, input_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        input_mask = matrix[:, :3, 131:]
        #torch.Size([32, 24, 129])
        output_matrix = matrix[:, 3:, :131]
        #torch.Size([32, 24, 129])
        output_mark = torch.arange(0, output_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        output_mask = matrix[:, 3:, 131:]
        #torch.Size([32, 24, 129])
        dec_inp = torch.zeros_like(output_matrix[:, -args.pred_len:, :]).float()
        #print(dec_inp.size())
        #dec_inp = torch.cat([output_matrix[:, :args.label_len, :], dec_inp], dim=1).float().cuda()
        #raise Exception
        #print(repr(model))
        if 'Linear' not in str(type(model)) and "Autoregressive" in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark, trainn=False)#, enc_self_mask=input_mask, dec_self_mask=output_mask)
        elif "Linear" not in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark)#, enc_self_mask=input_mask, dec_self_mask=output_mask)

        else:
            output = model(input_matrix)[:, :, :131]
        loss = output_mask[:, -args.pred_len:, :]*(
        output-output_matrix[:, -args.pred_len:, :])**2
        loss_list.extend(loss.sum(axis=-1).mean(axis=-1).detach().cpu().tolist())
        loss = loss.sum(axis=-1).mean()
        val_loss += loss.detach().cpu()
        pbar.set_description('%f' % val_loss)
    loss_p = e_loss*batch_size/samples_per_epoch
    val_loss_p = val_loss*batch_size/len(fore_valid_op)
    print('Epoch', e, 'loss', loss_p, 'val loss', val_loss_p, "mean and std", np.mean(loss_list), np.std(loss_list))
    with open('loss_values_log', 'a') as f:
        f.write(str(e)+' ' + str(loss_p.item()) + ' ' + str(val_loss_p.item())+ '\n')
    scheduler.step(val_loss_p.item())
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), fore_savepath)
        best_epoch = e
    if (e-best_epoch) > patience:
        break
    
print('Training has ended.')

#Informer IMS, Sampling, Backprop


# In[ ]:


V_reduced = 6
#Michi: die ganzen Argparses könnte man noch anders implementieren, wollte es nur schnell zum laufen bringen.
import argparse
parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model', type=str, required=False, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=False, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=3, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=3, help='prediction sequence length')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers 
#Michi: bisher nur default=3 zum laufen gebracht
parser.add_argument('--embed_type', type=int, default=3, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=V_reduced*2, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=V_reduced, help='decoder input size')
parser.add_argument('--c_out', type=int, default=V_reduced, help='output size')
parser.add_argument('--d_model', type=int, default=50, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=False)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args(args=[])
import importlib


import matplotlib.pyplot as plt


# In[ ]:


import models.InformerAutoregressiveFull as autoformer
importlib.reload(autoformer)
model = autoformer.Model(args).cuda()

lr, batch_size, samples_per_epoch, patience = 0.0005, 32, int(102400/sample_divisor), 6
d, N, he, dropout = 50, 2, 4, 0.2
V=131
print('number of parameters: ', V)


factor_indices1 = [var_to_ind[x]-1 for x in ["HR", "RR", "SBP", "DBP", "MBP", "O2 Saturation"]]
factor_indices2 = factor_indices1 + [V+x for x in factor_indices1]

#a = summary(model, [(32, 2), (32, 880), (32, 880), (32, 880)],  # shape of fore_train_ip
#            dtypes=[torch.float, torch.float, torch.float, torch.long])
#print(a)  # Model summary
# raise Exception
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
# Pretrain fore_model.
best_val_loss = np.inf
N_fore = len(fore_train_op)
#print(N_fore)
fore_savepath = 'informer_mimic_reduced_3hours.pytorch'
loss_func = torch.nn.MSELoss(reduction="none")

# torch.compile(model)
for e in range(number_of_epochs):
    e_indices = np.random.choice(range(N_fore), size=samples_per_epoch, replace=False)
    e_loss = 0
    pbar = tqdm(range(0, len(e_indices), batch_size))
    model.train()
    for start in pbar:
        ind = e_indices[start:start+batch_size]

        matrix = torch.tensor(fore_train_op[ind], dtype=torch.float32).cuda()[:,:,factor_indices2]
        #torch.Size([32, 48, 258])
        input_matrix = matrix[:, :3, :V_reduced*2]
        #torch.Size([32, 24, 129])
        input_mark = torch.arange(0, input_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        input_mask = matrix[:, :3, V_reduced:]
        #torch.Size([32, 24, 129])
        output_matrix = matrix[:, 3:, :V_reduced]
        #torch.Size([32, 24, 129])
        
        output_mark = torch.arange(0, output_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        output_mask = matrix[:, 3:, V_reduced:]
        #torch.Size([32, 24, 129])
        dec_inp = torch.zeros_like(output_matrix[:, -args.pred_len:, :]).float()
        #print(dec_inp.size())
        #dec_inp = torch.cat([output_matrix[:, :args.label_len, :], dec_inp], dim=1).float().cuda()
        #raise Exception
        #print(str(type(model)))
        #print(model)
        if 'Linear' not in str(type(model)) and "Autoregressive" in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark, tgt=output_matrix, trainn=False, backprop=True)#, enc_self_mask=input_mask, dec_self_mask=output_mask)
        elif "Linear" not in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark)#, enc_self_mask=input_mask, dec_self_mask=output_mask)

        else:
            output = model(input_matrix)[:, :, :V_reduced]
        loss = output_mask[:, -args.pred_len:, :]*(
        output-output_matrix[:, -args.pred_len:, :])**2
        loss = loss.sum(axis=-1).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        e_loss += loss.detach()
        pbar.set_description('%f' % e_loss)
    val_loss = 0
    #raise Exception
    loss_list = []
    model.eval()
    pbar = tqdm(range(0, len(fore_valid_op), 32))  # len(fore_valid_op)           ####################   maybe also batch_size instead of 32
    for start in pbar:
        matrix = torch.tensor(fore_valid_op[start:start+batch_size], dtype=torch.float32).cuda()[:, :, factor_indices2]
        #torch.Size([32, 48, 258])
        input_matrix = matrix[:, :3, :V_reduced*2]
        #torch.Size([32, 24, 129])
        input_mark = torch.arange(0, input_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        input_mask = matrix[:, :3, V_reduced:]
        #torch.Size([32, 24, 129])
        output_matrix = matrix[:, 3:, :V_reduced]
        #torch.Size([32, 24, 129])
        output_mark = torch.arange(0, output_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        output_mask = matrix[:, 3:, V_reduced:]
        #torch.Size([32, 24, 129])
        dec_inp = torch.zeros_like(output_matrix[:, -args.pred_len:, :]).float()
        #print(dec_inp.size())
        #dec_inp = torch.cat([output_matrix[:, :args.label_len, :], dec_inp], dim=1).float().cuda()
        #raise Exception
        #print(repr(model))
        if 'Linear' not in str(type(model)) and "Autoregressive" in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark, trainn=False)#, enc_self_mask=input_mask, dec_self_mask=output_mask)
        elif "Linear" not in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark)#, enc_self_mask=input_mask, dec_self_mask=output_mask)

        else:
            output = model(input_matrix)[:, :, :V_reduced]
        loss = output_mask[:, -args.pred_len:, :]*(
        output-output_matrix[:, -args.pred_len:, :])**2
        loss_list.extend(loss.sum(axis=-1).mean(axis=-1).detach().cpu().tolist())
        loss = loss.sum(axis=-1).mean()
        val_loss += loss.detach().cpu()
        pbar.set_description('%f' % val_loss)
    loss_p = e_loss*batch_size/samples_per_epoch
    val_loss_p = val_loss*batch_size/len(fore_valid_op)
    print('Epoch', e, 'loss', loss_p, 'val loss', val_loss_p, "mean and std", np.mean(loss_list), np.std(loss_list))
    with open('loss_values_log', 'a') as f:
        f.write(str(e)+' ' + str(loss_p.item()) + ' ' + str(val_loss_p.item())+ '\n')
    scheduler.step(val_loss_p.item())
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), fore_savepath)
        best_epoch = e
    if (e-best_epoch) > patience:
        break
    
print('Training has ended.')

#Informer IMS, Sampling, Backprop


# In[ ]:




