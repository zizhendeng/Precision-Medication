
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

from typing import List, Any
from typing import Optional

import SearchAdapter

import LoadData
import random


# data: [0][step][key]
def getInitState(predictor: L.Trainer, en_model:LitCRN,
                 data, headers: List[str],
                 num_treatments : int,
                 rnn_hidden_units: int,
                 encode_steps):
    
    max_seq_len = LoadData.getMaxSequenceLength(data)
    
    if encode_steps < max_seq_len:
        tmp_data = data
        data = []
        data.append([])
        for i in range(encode_steps):
            data[0].append(tmp_data[0][i])

    encoder_data = LoadData.getEncoderData(data, headers,
                                           LoadData.getMaxSequenceLength(data),
                                           num_treatments)
    encoder_dataset = CRNDataset(encoder_data)

    predictions = predictor.predict(en_model, DataLoader(encoder_dataset, batch_size=1))

    length = encoder_data["sequence_lengths"][0]

    br = predictions[0][0][0][length-1]
    assert len(br.shape) == 1
    hd = br.repeat(int(rnn_hidden_units/br.shape[0]))
    hd = hd.reshape((1,1)+hd.shape)
    cell = torch.zeros_like(hd)

    hx = (hd, cell)
    V = encoder_data['inputV'][0][length-1]
    output = encoder_data['outputs'][0][length-1]

    return hx, V, output

# K, C are Lipschitz constants
# see the paper for details
def getLVals(K:float, C: float, timesteps:int):
    ans = np.zeros(timesteps, dtype=np.float32)
    ans[timesteps-1] = C
    for i in range(timesteps-2, -1, -1):
        # warning: this may be wrong
        ans[i] = C*K + ans[i+1]*K
    return ans

# warning: this may be wrong
def getReward(sofa: float) -> float:
    return 3.0 - sofa

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 设定所有GPU的随机种子
    torch.backends.cudnn.deterministic = True  # 使得卷积操作是确定性的
    torch.backends.cudnn.benchmark = False  # 防止选择最优算法，保持确定性

# warning: this may be wrong
def getfunc(de_trainer:L.Trainer, de_lit_model: LitCRN, num_treatments:int):

    def func(rnn_state, V, output, treatment) -> tuple[Any, Any, Any, Any]:
        treat = treatment.getval()
        set_seed()
        assert len(V.shape) == 1 and len(output.shape) == 1
        array_treat = np.zeros((1,1,num_treatments), dtype=np.float32)
        array_treat[0,0,treat] = 1.0
        covariates = np.concatenate([V, output], axis=0)
        covariates = covariates.reshape((1,1)+covariates.shape)
        data_map = {
            "init_states": [rnn_state],
            "previous_treatments": array_treat,
            "current_treatments": array_treat,
            "current_covariates": np.concatenate([covariates, array_treat], axis=-1),
            "outputs": output.reshape((1,1)+output.shape),
            "sequence_lengths": [1],
            "active_entries": [np.ones((1,1), dtype=np.float32)],
        }
        dataset = CRNDataset(data_map)
        result = de_trainer.predict(de_lit_model, DataLoader(dataset))
        _,_, next_out, next_hx = result[0]
        next_out = next_out.reshape(output.shape).numpy()

        assert next_out.shape[0]==1
        reward = getReward(next_out[0])
        return next_hx, V, next_out, reward
    
    return func

    # this one is pretty slow, I don't know why
    de_model :CRNModel = de_lit_model.crn_model
    def testfunc(rnn_state, V, output, treatment) -> tuple[Any, Any, Any, Any]:
        treat = treatment.getval()

        assert len(V.shape) == 1 and len(output.shape) == 1
        array_treat = np.zeros((1,1,num_treatments), dtype=np.float32)
        array_treat[0,0,treat] = 1.0        


        assert len(V.shape) == 1 and len(output.shape) == 1
        array_treat = np.zeros((1,1,num_treatments), dtype=np.float32)
        array_treat[0,0,treat] = 1.0
        covariates = np.concatenate([V, output], axis=0)
        covariates = covariates.reshape((1,1)+covariates.shape)

        device = de_lit_model.device
        covariates = torch.tensor(covariates).to(device)
        array_treat = torch.tensor(array_treat).to(device)


        _,_, next_out, next_hx = de_model.forward(covariates, array_treat, rnn_state, 0)

        # this code is ugly
        next_out = next_out.reshape(output.shape).cpu().detach().numpy()

        assert next_out.shape[0]==1
        reward = getReward(next_out[0])

        return next_hx, V, next_out, reward
    
    return testfunc
        

def getInitalTreatments(timestep:int)->List[SearchAdapter.CRNTreatment]:
    ans = []
    for i in range(timestep):
        ans.append(SearchAdapter.CRNTreatment(0))
    return ans

def getAllTreatments(num_treatments:int)->List[SearchAdapter.CRNTreatment]:
    ans = []
    for i in range(num_treatments):
        ans.append(SearchAdapter.CRNTreatment(i))
    return ans

# data: [0][step][key]
# K and C are Lipschitz constants, see the paper for details
def getResult(predictor : L.Trainer, en_model : LitCRN,
              de_trainer: L.Trainer, de_model : LitCRN,
              data, headers: List[str],
              num_treatments:int, timesteps:int,
              rnn_hidden_units: int,
              K: float, C:float,encode_steps,
              MForEstimator:int=10) -> Optional[tuple[float, SearchAdapter.State, List[SearchAdapter.Treatment]]]:
    init_rnn_state, init_V, init_output = getInitState(predictor, en_model, data, headers,
                                                       num_treatments, rnn_hidden_units, encode_steps)
    LVals = getLVals(K, C, timesteps)
    func = getfunc(de_trainer, de_model, num_treatments)
    return SearchAdapter.getResult(init_rnn_state, init_V, init_output,
                                   LVals, func,
                                   getInitalTreatments(timesteps),
                                   getAllTreatments(num_treatments),
                                   MForEstimator,
                                   timesteps, timesteps)

