from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import logging
import os
import pickle
import numpy as np

from os import path
from typing import Optional

from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.mnist_datamodule import MNIST
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from CRN_model import CRNModel, CRNDataset




class LitCRN(LightningModule):
    
    # params, hyperparams, b_train_decoder are used to initialize the  CRN model
    def __init__(self, params, hyperparams, b_train_decoder, learning_rate=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.crn_model = CRNModel(params, hyperparams, b_train_decoder)
    
    # def forward(self, current_covariates, previous_treatments, current_treatments):
    #     return self.crn_model(current_covariates, previous_treatments, current_treatments)
    

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")


    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        init_states = None
        if len(batch)==5:
            current_covariates,\
            target_treatments, target_outcomes,\
            active_entries, init_states = batch
        else:
            current_covariates,\
            target_treatments, target_outcomes,\
            active_entries = batch
        return self.crn_model(
            current_covariates, target_treatments, init_states, 0
        )

    def configure_optimizers(self):
        print(self.hparams.learning_rate)
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    

    def _common_step(self, batch, batch_idx, stage: str):
        if(self.trainer.max_epochs==-1):
            p = min(float(self.current_epoch)/100,1.0)
        else:
            p = float(self.current_epoch) / float(self.trainer.max_epochs)
        alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1
        init_states = None
        if len(batch)==5:
            current_covariates,\
            target_treatments, target_outcomes,\
            active_entries, init_states = batch
        else:
            current_covariates,\
            target_treatments, target_outcomes,\
            active_entries = batch
        _, treatment_predictions, outcome_predictions, _ = self.crn_model(
            current_covariates, target_treatments, init_states, alpha
        )
        loss, treatment_loss, outcome_loss = self.crn_model.compute_loss(
            treatment_predictions,
            outcome_predictions,
            target_treatments,
            target_outcomes,
            active_entries,
        )
        self.log(f"{stage}_loss", loss, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_treatment_loss", treatment_loss, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_outcome_loss", outcome_loss, on_epoch=True, sync_dist=True)
        return loss

class LitCRNDataModule(LightningDataModule):
    def __init__(self,
                 batch_size: int = 32,
                 train_dataset: Optional[CRNDataset] = None,
                 val_dataset: Optional[CRNDataset] = None,
                 test_dataset: Optional[CRNDataset] = None,
                 predict_dataset: Optional[CRNDataset] = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size
        

    def train_dataloader(self):
        assert self.train_dataset is not None
        return DataLoader(self.train_dataset, self.batch_size)

    def val_dataloader(self):
        assert self.val_dataset is not None
        return DataLoader(self.val_dataset, self.batch_size)

    def test_dataloader(self):
        assert self.test_dataset is not None
        return DataLoader(self.test_dataset, self.batch_size)

    def predict_dataloader(self):
        assert self.predict_dataset is not None
        return DataLoader(self.predict_dataset, self.batch_size)
