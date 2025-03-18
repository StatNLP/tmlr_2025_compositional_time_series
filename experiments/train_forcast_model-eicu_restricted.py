#!/usr/bin/env python

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
tqdm.pandas()
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau


import argparse
parser2 = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
parser2.add_argument('--gold_path_train', type=str, default='', help='root path of the data file')
parser2.add_argument('--gold_path_val', type=str, default='', help='root path of the data file')
parser2.add_argument('--silver_path', type=str, default="", help='root path of the data file')
parser2.add_argument('--silver_only', type=str, default="True", help='perform training only on silver')
parser2.add_argument('--pretrained_model', type=str, default="", help='load a pretrained model')
parser2.add_argument('--savedir', type=str, default="", help='savedir')
parser2.add_argument('--gold_only', type=str, default="False", help='perform training only on gold')
parser2.add_argument('--silver_amount', type=int, default=246080000, help="how much silver data to include")
parser2.add_argument('--number', type=int, default=1, help="which try is it")
args2 = parser2.parse_args()


if args2.gold_path_train == args2.gold_path_val:
    with open(args2.gold_path_val, "rb") as pickle_file:
        fore_train_op, fore_valid_op=joblib.load(pickle_file)
else:
    with open(args2.gold_path_val, "rb") as pickle_file:
        fore_valid_op=joblib.load(pickle_file) 
    with open(args2.gold_path_train, "rb") as pickle_file:
        fore_train_op=joblib.load(pickle_file)

if args2.silver_only == "True":
    fore_train_op = None
    
    
with open(args2.silver_path, "rb") as pickle_file:
    fore_train_op_silver=joblib.load(pickle_file)
    fore_train_op_silver=fore_train_op_silver[:args2.silver_amount]


if args2.silver_only == "True":
    fore_train_op = fore_train_op_silver
elif args2.gold_only == "True":
    pass
else:
    fore_train_op = np.concatenate([fore_train_op, fore_train_op_silver])
print(fore_train_op.shape)
sample_divisor=1
number_of_epochs = 1000


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
parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers 

parser.add_argument('--embed_type', type=int, default=3, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=12, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=6, help='decoder input size')
parser.add_argument('--c_out', type=int, default=6, help='output size')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
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


import models.InformerAutoregressiveFull as autoformer
importlib.reload(autoformer)
model = autoformer.Model(args).cuda()

if args2.pretrained_model:
    model.load_state_dict(torch.load(args2.pretrained_model))
lr, batch_size, samples_per_epoch, patience = 0.0005, 32, int(102400/sample_divisor), 6


samples_per_epoch = min((samples_per_epoch, fore_train_op.shape[0]))
d, N, he, dropout = 50, 2, 4, 0.2
V=131

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
# Pretrain fore_model.
best_val_loss = np.inf
N_fore = len(fore_train_op)
fore_savepath = args2.savedir+args2.silver_path.split("/")[-1].split(".")[0]+"-SILVER_ONLY_"+str(args2.silver_only)+"_n_"+str(args2.silver_amount)+"_"+str(args2.number)+"_PRETRAINED_MODEL_" + (args2.pretrained_model.split("/")[-1] if args2.pretrained_model else "NONE") + ".model"
print(fore_savepath)
loss_func = torch.nn.MSELoss(reduction="none")


for e in range(number_of_epochs):
    e_indices = np.random.choice(range(N_fore), size=samples_per_epoch, replace=False)
    e_loss = 0
    pbar = tqdm(range(0, len(e_indices), batch_size))
    model.train()
    for start in pbar:
        ind = e_indices[start:start+batch_size]
        matrix = torch.tensor(fore_train_op[ind], dtype=torch.float32).cuda()
        input_matrix = matrix[:, :24, :12]
        input_mark = torch.arange(0, input_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        input_mask = matrix[:, :24, 6:]
        output_matrix = matrix[:, 24:, :6]
        output_mark = torch.arange(0, output_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        output_mask = matrix[:, 24:, 6:]
        dec_inp = torch.zeros_like(output_matrix[:, -args.pred_len:, :]).float()
        if 'Linear' not in str(type(model)) and "Autoregressive" in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark, tgt=output_matrix, trainn=False, backprop=True)
        elif "Linear" not in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark)
        else:
            output = model(input_matrix)[:, :, :6]
        loss = output_mask[:, -args.pred_len:, :]*(
        output-output_matrix[:, -args.pred_len:, :])**2
        loss = loss.sum(axis=-1).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        e_loss += loss.detach()
        pbar.set_description('%f' % e_loss)
    val_loss = 0
    loss_list = []
    model.eval()
    pbar = tqdm(range(0, len(fore_valid_op), 32))
    for start in pbar:
        matrix = torch.tensor(fore_valid_op[start:start+batch_size], dtype=torch.float32).cuda()
        input_matrix = matrix[:, :24, :12]
        input_mark = torch.arange(0, input_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        input_mask = matrix[:, :24, 6:]
        output_matrix = matrix[:, 24:, :6]
        output_mark = torch.arange(0, output_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        output_mask = matrix[:, 24:, 6:]
        dec_inp = torch.zeros_like(output_matrix[:, -args.pred_len:, :]).float()
        print(input_matrix.size())
        if 'Linear' not in str(type(model)) and "Autoregressive" in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark, trainn=False)
        elif "Linear" not in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark)
        else:
            output = model(input_matrix)[:, :, :6]
        loss = output_mask[:, -args.pred_len:, :]*(
        output-output_matrix[:, -args.pred_len:, :])**2
        loss_list.extend(loss.sum(axis=-1).mean(axis=-1).detach().cpu().tolist())
        loss = loss.sum(axis=-1).mean()
        val_loss += loss.detach().cpu()
        pbar.set_description('%f' % val_loss)
    loss_p = e_loss*batch_size/samples_per_epoch
    val_loss_p = val_loss*batch_size/len(fore_valid_op)
    print('Epoch', e, 'loss', loss_p, 'val loss', val_loss_p, "mean and std", np.mean(loss_list), np.std(loss_list))
    with open(fore_savepath+".txt", 'a') as f:
        f.write(str(e)+' ' + str(loss_p.item()) + ' ' + str(val_loss_p.item())+ '\n')
    scheduler.step(val_loss_p.item())
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), fore_savepath)
        best_epoch = e
    if (e-best_epoch) > patience:
        break
    
print('Training has ended.')