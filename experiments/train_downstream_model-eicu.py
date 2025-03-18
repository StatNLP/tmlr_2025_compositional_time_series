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

# In[5]:

print("test123")

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


# In[22]:


import argparse
parser2 = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
#parser2.add_argument('--gold_path', type=str, default='very_important_blub.pkl', help='root path of the data file')
parser2.add_argument('--gold_path_train', type=str, default='/home/mitarb/hagmann/projects/structures_compositionality/gold_data/mimic-full/gold_train.pickle', help='root path of the data file')
#parser2.add_argument('--gold_path_val', type=str, default='very_important_blub.pkl', help='root path of the data file')
parser2.add_argument('--gold_path_val', type=str, default='/home/mitarb/hagmann/projects/structures_compositionality/gold_data/mimic-full/gold_valid.pickle', help='root path of the data file')
parser2.add_argument('--silver_path', type=str, default="/home/mitarb/hagmann/projects/structures_compositionality/synthetic_inputs/embd_geca.pickle", help='root path of the data file')
parser2.add_argument('--silver_only', type=str, default="True", help='perform training only on silver')
parser2.add_argument('--pretrained_model', type=str, default="", help='load a pretrained model')
parser2.add_argument('--savedir', type=str, default="results-28May/", help='savedir')
parser2.add_argument('--gold_only', type=str, default="False", help='perform training only on gold')
parser2.add_argument('--silver_amount', type=int, default=246080, help="how much silver data to include")
parser2.add_argument('--number', type=int, default=1, help="which try is it")
parser2.add_argument('--partial', type=float, default=1.0)
args2 = parser2.parse_args()


# ## Load forecast dataset into matrices.

# In[7]:


if args2.gold_path_train == args2.gold_path_val:
    with open(args2.gold_path_val, "rb") as pickle_file:
        fore_train_op, fore_valid_op=joblib.load(pickle_file)
else:
    with open(args2.gold_path_val, "rb") as pickle_file:
        #fore_train_op, fore_valid_op=joblib.load(pickle_file) #change here if use non-sliding window data
        fore_valid_op=joblib.load(pickle_file) #change here if use non-sliding window data
    with open(args2.gold_path_train, "rb") as pickle_file:
        fore_train_op=joblib.load(pickle_file)

if args2.silver_only == "True":
    fore_train_op = None
    


with open(args2.silver_path, "rb") as pickle_file:
    fore_train_op_silver=joblib.load(pickle_file)
    fore_train_op_silver=fore_train_op_silver[:args2.silver_amount]



print(fore_train_op_silver.shape)


# In[8]:


if args2.silver_only == "True":
    fore_train_op = fore_train_op_silver
elif args2.gold_only == "True":
    pass
else:
    fore_train_op = np.concatenate([fore_train_op, fore_train_op_silver])
print(fore_train_op.shape)
sample_divisor=1
number_of_epochs = 1000


import os
if os.path.exists("mimic_indices.npy"):
    indices = np.load("mimic_indices.npy")
else:
    indices = np.array(list(range(len(fore_train_op))))
    np.random.shuffle(indices)
    np.save("mimic_indices.npy", indices)

length = int(len(indices) * args2.partial)
new_indices = indices[:length]

fore_train_op = fore_train_op[new_indices]


# In[9]:

mean_std_dict = pickle.load(open("/home/mitarb/hagmann/projects/llm_circ/imputed_data/mimic-full/mean_std_dict.pickle", "rb"))
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
parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers 
#Michi: bisher nur default=3 zum laufen gebracht
parser.add_argument('--embed_type', type=int, default=3, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=262, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=131, help='decoder input size')
parser.add_argument('--c_out', type=int, default=131, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
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

var_to_ind= {'ALP': 1, 'ALT': 2, 'AST': 3, 'Albumin': 4, 'Albumin 25%': 5, 'Albumin 5%': 6,
              'Amiodarone': 7, 'Anion Gap': 8, 'Antibiotics': 9, 'BUN': 10, 'Base Excess': 11,
              'Basophils': 12, 'Bicarbonate': 13, 'Bilirubin (Direct)': 14, 'Bilirubin (Indirect)': 15,
              'Bilirubin (Total)': 16, 'CRR': 17, 'Calcium Free': 18, 'Calcium Gluconate': 19,
              'Calcium Total': 20, 'Cefazolin': 21, 'Chest Tube': 22, 'Chloride': 23, 'Colloid': 24,
              'Creatinine Blood': 25, 'Creatinine Urine': 26, 'D5W': 27, 'DBP': 28, 'Dextrose Other': 29,
              'Dobutamine': 30, 'Dopamine': 31, 'EBL': 32, 'Emesis': 33, 'Eoisinophils': 34,
              'Epinephrine': 35, 'Famotidine': 36, 'Fentanyl': 37, 'FiO2': 38, 'Fiber': 39,
              'Free Water': 40, 'Fresh Frozen Plasma': 41, 'Furosemide': 42, 'GCS_eye': 43,
              'GCS_motor': 44, 'GCS_verbal': 45, 'GT Flush': 46, 'Gastric': 47, 'Gastric Meds': 48,
              'Glucose (Blood)': 49, 'Glucose (Serum)': 50, 'Glucose (Whole Blood)': 51,
              'HR': 52, 'Half Normal Saline': 53, 'Hct': 54, 'Heparin': 55, 'Hgb': 56,
              'Hydralazine': 57, 'Hydromorphone': 58, 'INR': 59, 'Insulin Humalog': 60,
              'Insulin NPH': 61, 'Insulin Regular': 62, 'Insulin largine': 63,
              'Intubated': 64, 'Jackson-Pratt': 65, 'KCl': 66, 'KCl (Bolus)': 67,
              'LDH': 68, 'Lactate': 69, 'Lactated Ringers': 70, 'Levofloxacin': 71,
              'Lorazepam': 72, 'Lymphocytes': 73, 'Lymphocytes (Absolute)': 74,
              'MBP': 75, 'MCH': 76, 'MCHC': 77, 'MCV': 78, 'Magnesium': 79,
              'Magnesium Sulfate (Bolus)': 80,  'Magnesium Sulphate': 81,
              'Mechanically ventilated': 82, 'Metoprolol': 83, 'Midazolam': 84,
              'Milrinone': 85, 'Monocytes': 86, 'Morphine Sulfate': 87,
              'Neosynephrine': 88, 'Neutrophils': 89, 'Nitroglycerine': 90,
              'Nitroprusside': 91, 'Norepinephrine': 92, 'Normal Saline': 93,
              'O2 Saturation': 94, 'OR/PACU Crystalloid': 95, 'PCO2': 96,
              'PO intake': 97, 'PO2': 98, 'PT': 99, 'PTT': 100, 'Packed RBC': 101,
              'Pantoprazole': 102, 'Phosphate': 103, 'Piggyback': 104, 'Piperacillin': 105,
              'Platelet Count': 106, 'Potassium': 107, 'Pre-admission Intake': 108,
              'Pre-admission Output': 109, 'Propofol': 110, 'RBC': 111, 'RDW': 112,
              'RR': 113, 'Residual': 114, 'SBP': 115, 'SG Urine': 116, 'Sodium': 117,
              'Solution': 118, 'Sterile Water': 119, 'Stool': 120, 'TPN': 121,
              'Temperature': 122, 'Total CO2': 123, 'Ultrafiltrate': 124, 'Urine': 125,
              'Vancomycin': 126, 'Vasopressin': 127, 'WBC': 128, 'Weight': 129,
              'pH Blood': 130, 'pH Urine': 131}

# In[25]:

def get_sofa(matrix, var_to_ind): #24x131 matrix
    # GCS: min_eye, min_motor, min_verbal = 5, 5, 5
    #raise Exception
    key ="GCS_eye"
    var_to_ind = {x:i-1 for x,i in var_to_ind.items()}
    a=matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]
    min_eye = min(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=4)
    key = "GCS_motor"
    min_motor = min(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=6)
    key = "GCS_verbal"
    min_verbal = min(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=5)
    

    GCS = min_eye + min_motor + min_verbal
    if GCS > 14: GCS_sofa = 0
    elif GCS > 12: GCS_sofa = 1
    elif GCS > 9:  GCS_sofa = 2
    elif GCS > 5:  GCS_sofa = 3
    else: GCS_sofa = 4
    #print('GCS_sofa is', GCS_sofa, ';     GCS is', GCS,'; GCS eye', min_eye, '; GCS motor', min_motor, '; GCS verbal', min_verbal)

    key = "Bilirubin (Total)"
    bilir = max(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=1)
    if bilir > 12: bilir_sofa = 4
    elif bilir > 6: bilir_sofa = 3
    elif bilir > 2: bilir_sofa = 2
    elif bilir > 1.2: bilir_sofa = 1
    else: bilir_sofa = 0
    #print('bilir_Sofa is', bilir_sofa, ';   bilirubin is', bilir)
    
    # Coagulation (Platelets)
    key = "Platelet Count"
    plate = max(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=160)
    if plate > 150: plate_sofa = 0
    elif plate > 100: plate_sofa = 1
    elif plate > 50: plate_sofa = 2
    elif plate > 20: plate_sofa = 3
    else: plate_sofa = 4
    #print('plate_sofa is', plate_sofa, ';   platelet count is', plate)
    
    # print('Urinmenge 24h', sum(data_var[data_var['variable']=='Urine']['value2']))

    key = "Urine"
    urine = sum(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0])
    key = "Creatinine Blood"
    creat = max(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=1)
    
    if (urine < 200) or (creat > 5): renal_sofa = 4
    elif  (urine < 500) or (creat > 3.5): renal_sofa = 3
    elif creat > 2.0: renal_sofa = 2
    elif creat > 1.2: renal_sofa = 1
    else: renal_sofa = 0
    #print('renal_sofa:',renal_sofa,';       urine 24:',urine,'; creat:', creat)
    
    CS_data = get_CS(matrix, var_to_ind)
    cs_sofa = CS_SOFA(CS_data)
    
    #cs_sofa = 0
    key="FiO2"
    fio2 = (matrix[:, var_to_ind[key]]*mean_std_dict[key][1]+mean_std_dict[key][0])
    key="PO2"
    po2 = (matrix[:, var_to_ind[key]]*mean_std_dict[key][1]+mean_std_dict[key][0])
    PaO2FiO2 = 100*po2/fio2
    PaO2FiO2 = PaO2FiO2[torch.nonzero(PaO2FiO2, as_tuple=True)]
    pao2fio2 = min(PaO2FiO2)
    if pao2fio2<100: resp=4
    elif pao2fio2<200: resp=3
    elif pao2fio2<300:resp=2
    elif pao2fio2<400:resp=1
    else: resp=0
    return GCS_sofa, cs_sofa, resp, plate_sofa, bilir_sofa, renal_sofa

def get_CS(matrix, var_to_ind):
    #data_var = data_pat[data_pat['variable'].isin(['Dobutamine','Dopamine','Epinephrine','Norepinephrine','Weight'])]
    #data_var['value2'] = data_var['value']*data_var['std']+data_var['mean']
    
    #weight = min(data_var[data_var['variable']=='Weight']['value2'], default=80)  # set default weight to 80kg.
    key = "Weight"
    weight = min(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)], default=80)#*mean_std_dict[key][1]+mean_std_dict[key][0], default=1)
    key = "Dopamine"
    try:
        data_dop = max(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=1)
        data_dop = data_dop /60/weight*1000
    except:
        data_dop = 0

    key = "Dobutamine"
    
    try:
        data_dobu = max(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=1)
        data_dobu = data_dobu  /60/weight*1000
    except:
        data_dobu = 0
    key = "Epinephrine"
    
    try:
        data_epi = max(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=1)
        data_epi = data_epi  /60/weight*1000
    except:
        data_epi = 0
    key = "Norepinephrine"
    
    try:
        data_nore = max(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=1)
        data_nore = data_nore /60/weight*1000
    except:
        data_nore = 0
        


    key = "SBP"
    SBP = (matrix[:, var_to_ind[key]]*mean_std_dict[key][1]+mean_std_dict[key][0])

    key = "DBP"
    DBP = (matrix[:, var_to_ind[key]]*mean_std_dict[key][1]+mean_std_dict[key][0])

    MAP = 2/3 * DBP + 1/3 * SBP
    MAP = min(MAP[torch.nonzero(MAP, as_tuple=True)], default=100)
                 
        
    return MAP, data_dop, data_dobu, data_epi, data_nore 
    
def CS_SOFA(data):
    map = data[0]
    dop, dobu, epi, nore = data[1:5]
    # print('CS data: mdden', data)
    if (dop > 15) or (epi > 0.1) or (nore > 0.01): CS = 4
    elif (dop > 5) or (epi > 0) or (nore > 0): CS = 3
    elif (dop > 0) or (dobu > 0): CS = 2
    elif map < 70: CS = 1
    else: CS = 0
    # print('CS Sofa is:', CS)
    return CS 

import models.InformerAutoregressiveFullRegression as autoformer
importlib.reload(autoformer)
model = autoformer.Model(args).cuda()

if args2.pretrained_model:
    model.load_state_dict(torch.load(args2.pretrained_model))
lr, batch_size, samples_per_epoch, patience = 0.0005, 32, int(102400/sample_divisor), 6
#lr, batch_size, samples_per_epoch, patience = 0.0001, 32, int(102400/sample_divisor), 6

samples_per_epoch = min((samples_per_epoch, fore_train_op.shape[0]))
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
#print(N_fore)2
#fore_savepath = "results-28May/"
fore_savepath = args2.savedir+args2.silver_path.split("/")[-1].split(".")[0]+"-SILVER_ONLY_"+str(args2.silver_only)+"_n_"+str(args2.silver_amount)+"_"+str(args2.number)+"_PRETRAINED_MODEL_" + (args2.pretrained_model.split("/")[-1] if args2.pretrained_model else "NONE") + str(args2.partial)+".model"
print(fore_savepath)
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
        input_matrix = matrix[:, :24, :262]
        #torch.Size([32, 24, 129])
        input_mark = torch.arange(0, input_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        input_mask = matrix[:, :24, 131:]
        #torch.Size([32, 24, 129])
        output_matrix = matrix[:, 24:, :131]
        #torch.Size([32, 24, 129])
        
        output_mark = torch.arange(0, output_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        output_mask = matrix[:, 24:, 131:]
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
        loss = 0
        for outpu, real, mask, input_mat in zip(output, output_matrix, output_mask, input_matrix):
            b = sum(get_sofa(real, var_to_ind))
            loss_term = (outpu-b)**2
            loss+=loss_term
            #print(loss_term)
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
        input_matrix = matrix[:, :24, :262]
        #torch.Size([32, 24, 129])
        input_mark = torch.arange(0, input_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        input_mask = matrix[:, :24, 131:]
        #torch.Size([32, 24, 129])
        output_matrix = matrix[:, 24:, :131]
        #torch.Size([32, 24, 129])
        output_mark = torch.arange(0, output_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        output_mask = matrix[:, 24:, 131:]
        #torch.Size([32, 24, 129])
        dec_inp = torch.zeros_like(output_matrix[:, -args.pred_len:, :]).float()
        #print(dec_inp.size())
        #dec_inp = torch.cat([output_matrix[:, :args.label_len, :], dec_inp], dim=1).float().cuda()
        #raise Exception
        #print(repr(model))
        print(input_matrix.size())
        if 'Linear' not in str(type(model)) and "Autoregressive" in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark, trainn=False)#, enc_self_mask=input_mask, dec_self_mask=output_mask)
        elif "Linear" not in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark)#, enc_self_mask=input_mask, dec_self_mask=output_mask)

        else:
            output = model(input_matrix)[:, :, :131]
        loss = 0
        for outpu, real, mask, input_mat in zip(output, output_matrix, output_mask, input_matrix):
            b = sum(get_sofa(real, var_to_ind))
            loss_term = (outpu-b)**2
            loss+=loss_term
            #print(loss_term)
        loss.backward()
        val_loss += loss.detach().cpu()
        pbar.set_description('%f' % val_loss)
    loss_p = e_loss*batch_size/samples_per_epoch
    val_loss_p = val_loss*batch_size/len(fore_valid_op)
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

#Informer IMS, Sampling, Backprop


# In[24]:





# In[ ]:




