
import numpy as np
import os

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import logging
import argparse

from CRN_model import CRNModel, CRNDataset
from CRN_Lightning_model import LitCRN, LitCRNDataModule
from lightning.pytorch.cli import LightningCLI
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger

import LoadData

import SearchApp

import pickle

import CRN_Lightning_model

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 设定所有GPU的随机种子
    torch.backends.cudnn.deterministic = True  # 使得卷积操作是确定性的
    torch.backends.cudnn.benchmark = False  # 防止选择最优算法，保持确定性
    
set_seed()


# Lipschitz constant, see the paper for details
lip_const_K = 1.0

b_train_new = False

b_new_encoder_data = False
b_new_decoder_data = False

devices = "1"

encoder_model_filename = './models/encoder.ckpt'
decoder_model_filename = "./models/decoder.ckpt"

encoder_data_filename = "./data/encoder.pkl"
decoder_data_filename = "./data/decoder.pkl"


csv_data, headers = LoadData.load('./data/truncated_sepsis_data_withTimes.csv')

if b_new_encoder_data:

    encoder_data = LoadData.getEncoderData(csv_data, headers,
                                        LoadData.getMaxSequenceLength(csv_data),
                                        LoadData.getMaxTreatmentNumber(csv_data, headers))
    with open(encoder_data_filename, "wb") as f:
        pickle.dump(encoder_data, f)
else:
    with open(encoder_data_filename, "rb") as f:
        encoder_data = pickle.load(f)

num_treatments = encoder_data['current_treatments'].shape[2]
num_covariates = encoder_data['current_covariates'].shape[2]
num_outputs = encoder_data['outputs'].shape[2]
length = encoder_data['current_treatments'].shape[1]

params = {
    "num_treatments": num_treatments,
    "num_covariates": num_covariates,
    "num_outputs": num_outputs,
    "max_sequence_length": length,
}
best_hyperparams = {
    "rnn_hidden_units": 96,
    "br_size": 24,
    "fc_hidden_units": 72,
    "rnn_keep_prob": 0.9,
    "lip_const_K": None
}

all_dataset = CRNDataset(encoder_data)

train_size = int(0.8 * len(all_dataset))
val_size = len(all_dataset) - train_size

# 随机分割数据集
training_set, validation_set = random_split(all_dataset, [train_size, val_size])


en_lit_crn = LitCRN(params,best_hyperparams, False, learning_rate=0.01)



# en_trainer = L.Trainer(max_epochs=100, devices=devices, logger=TensorBoardLogger("encoder_logs"))
en_trainer = L.Trainer(max_epochs=1, deterministic=True,   devices=devices)

if b_train_new:
    en_trainer.fit(en_lit_crn, DataLoader(training_set, batch_size=2), DataLoader(validation_set, batch_size=2))
else:
    en_lit_crn = LitCRN.load_from_checkpoint(encoder_model_filename)


predictor = L.Trainer(devices=devices,deterministic=True,   logger=False)

if b_new_decoder_data:
    predictions = predictor.predict(en_lit_crn, DataLoader(all_dataset))


    all_br = [a[0] for a in predictions]

    all_br = torch.concatenate(all_br, dim=0).numpy()

    seq_all_data = LoadData.getDecoderData(encoder_data, all_br, 5)
    
    with open(decoder_data_filename, "wb") as f:
        pickle.dump(seq_all_data, f)
else:
    with open(decoder_data_filename, "rb") as f:
        seq_all_data = pickle.load(f)

seq_all_set = CRNDataset(seq_all_data)

seq_training_size = int(0.8*len(seq_all_set))
seq_valid_size = len(seq_all_set) - seq_training_size


seq_training_set, seq_valid_set = random_split(seq_all_set, [seq_training_size, seq_valid_size])

length = seq_all_data['current_treatments'].shape[1]
num_covariates = seq_all_data['current_covariates'].shape[2]
params['max_sequence_length'] = length
params['num_covariates'] = num_covariates

# make sure Lipschitz continious for lstm
best_hyperparams["lip_const_K"] = lip_const_K

de_lit_model = LitCRN(params, best_hyperparams, True, 0.01)
# de_trainer = L.Trainer(max_epochs=10, devices=devices, logger=TensorBoardLogger("decoder_logs"))
de_trainer = L.Trainer(max_epochs=1,  deterministic=True, devices=devices)

if b_train_new:
    de_trainer.fit(de_lit_model, DataLoader(seq_training_set), DataLoader(seq_valid_set))
else:
    de_lit_model = LitCRN.load_from_checkpoint(decoder_model_filename)

results = SearchApp.getResult(predictor, en_lit_crn, L.Trainer(logger=False, deterministic=True,  devices=devices), de_lit_model,
                          [csv_data[0]], headers, num_treatments, 2,
                          best_hyperparams['rnn_hidden_units'],
                          lip_const_K, 1.0, 1, 10)

# node.gval, node.s, node.treatments, node.all_gvals
final_reward, final_state, all_treatments, all_rewards_history = results

for i in range(len(all_rewards_history)):
    if i == 0:
        last_rewards = 0
    else:
        last_rewards = all_rewards_history[i-1]
    reward_i = all_rewards_history[i] - last_rewards
    sofa_i = 3 - reward_i
    
    print(f"sofa value after ith treatment is : {sofa_i}")