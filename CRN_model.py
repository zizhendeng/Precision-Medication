import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import logging
import os
import pickle
import numpy as np

from utils.flip_gradient import FlipGradient

from LipDecoder import LipDecoder, LipLinear

class CRNModel(nn.Module):
    def __init__(
        self,
        params,
        hyperparams,
        b_train_decoder=False,
    ):
        super(CRNModel, self).__init__()
        self.num_treatments = params["num_treatments"]
        self.num_covariates = params["num_covariates"]
        self.num_outputs = params["num_outputs"]
        self.max_sequence_length = params["max_sequence_length"]

        self.br_size = hyperparams["br_size"]
        self.rnn_hidden_units = hyperparams["rnn_hidden_units"]
        self.fc_hidden_units = hyperparams["fc_hidden_units"]
        self.rnn_keep_prob = hyperparams["rnn_keep_prob"]
        self.rnn_lip_const = None
        if hyperparams["lip_const_K"] is not None:
            lip_const_K = hyperparams["lip_const_K"]
            lip_const_K_per_layer = lip_const_K**(1.0/4)
            self.rnn_lip_const = lip_const_K_per_layer/self.rnn_keep_prob
        else:
            lip_const_K_per_layer = None
            self.rnn_lip_const = None

        self.b_train_decoder = b_train_decoder

        if b_train_decoder:
            self.rnn_input = LipDecoder(
                input_dim=self.num_covariates,
                hidden_dim=self.rnn_hidden_units,
                lip_const=self.rnn_lip_const
            )
        else:
            self.rnn_input = nn.LSTM(
                input_size=self.num_covariates,
                hidden_size=self.rnn_hidden_units,
                num_layers=1,
                # dropout=1.0-self.rnn_keep_prob,
                batch_first=True,
            )

        
        self.br_layer = nn.Sequential(
            LipLinear(
                self.rnn_hidden_units, self.br_size, lip_const_K_per_layer
            ),
            nn.ELU(),
        )
        

        self.treatment_layer = nn.Sequential(
            nn.Linear(in_features=self.br_size, out_features=self.fc_hidden_units),
            nn.ELU(),
            nn.Linear(
                in_features=self.fc_hidden_units, out_features=self.num_treatments
            ),
        )

        self.outcome_layer = nn.Sequential(
            LipLinear(
                self.br_size + self.num_treatments,
                self.fc_hidden_units,
                lip_const_K_per_layer
            ),
            nn.ELU(),
            LipLinear(self.fc_hidden_units, self.num_outputs, lip_const_K_per_layer),
        )
    
    
    def forward_half(self, current_covariates, current_treatments, init_states, alpha):
        combined_input = current_covariates
        if self.b_train_decoder:
            # warning this may be wrong
            if type(init_states) == list:
                # init_state is hidden_state
                assert len(init_states) == 2
                [hd, cell] = init_states
                hx = (hd[0],cell[0])
            elif type(init_states) == tuple:
                # init_state is hidden_state
                assert len(init_states) == 2
                (hd, cell) = init_states
                hx = (hd,cell)                
            else:
                # init_state is balancing representation
                hd_state = init_states.repeat(1, int(self.rnn_hidden_units/init_states.shape[1]))
                hd_state = hd_state.reshape((1,)+hd_state.shape)
                hx = (hd_state, torch.zeros_like(hd_state))
            rnn_output, hx_output = self.rnn_input(combined_input, hx)
        else:
            rnn_output, hx_output = self.rnn_input(combined_input)
        current_treatments = current_treatments.reshape(
            -1, current_treatments.shape[-1]
        )
        batch_size = current_covariates.shape[0]

        # Flatten to apply same weights to all time steps.
        rnn_output = rnn_output.reshape(-1, self.rnn_hidden_units)
        rnn_output = nn.functional.dropout(rnn_output, p=1.0-self.rnn_keep_prob)

        balance_representation = self.br_layer(rnn_output)
        
        treatment_prob_predictions = torch.softmax(
            self.treatment_layer(FlipGradient.apply(balance_representation, alpha)), dim=-1
        )
        outcome_predictions = self.outcome_layer(
            torch.cat((balance_representation, current_treatments), dim=-1)
        )

        balance_representation = balance_representation.reshape(
            batch_size, -1, balance_representation.shape[-1]
        )
        treatment_prob_predictions = treatment_prob_predictions.reshape(
            batch_size, -1, treatment_prob_predictions.shape[-1]
        )
        outcome_predictions = outcome_predictions.reshape(
            batch_size, -1, outcome_predictions.shape[-1]
        )

        return balance_representation, treatment_prob_predictions, outcome_predictions, hx_output        

    def forward(self, current_covariates, current_treatments, init_states, alpha):
        if not self.b_train_decoder:
            return self.forward_half(current_covariates=current_covariates, current_treatments=current_treatments, init_states=init_states, alpha=alpha)
        seq_len = current_covariates.shape[1]
        bal_rep = None
        treat_prob = None
        out_pred = None
        hx_out = init_states
        for i in range(seq_len):
            if i == 0:
                bal_rep, treat_prob, out_pred, hx_out = self.forward_half(current_covariates[:,:1,:], current_treatments[:,:1,:], hx_out, alpha)
            else:
                rel_cov = current_covariates[:,i:i+1,:] + 0
                start_idx = self.num_covariates - self.num_outputs - self.num_treatments
                end_idx = self.num_covariates - self.num_treatments
                rel_cov[:,:,start_idx:end_idx] = out_pred[:,-1:,:]
                tmp_bal, tmp_treat, tmp_out, hx_out = self.forward_half(rel_cov, current_treatments[:,i:i+1,:], hx_out, alpha)
                bal_rep = torch.concatenate([bal_rep, tmp_bal], dim=1)
                treat_prob = torch.concatenate([treat_prob, tmp_treat], dim=1)
                out_pred = torch.concatenate([out_pred, tmp_out], dim=1)
        return bal_rep, treat_prob, out_pred, hx_out
            

    def compute_loss(
        self,
        treatment_predictions,
        outcome_predictions,
        target_treatments,
        target_outcomes,
        active_entries,
    ):
        treatment_loss = torch.sum(
            -torch.log(treatment_predictions+1e-8) * target_treatments * active_entries
        ) / (torch.sum(active_entries)+1)
        outcome_loss = torch.sum(
            (outcome_predictions - target_outcomes) ** 2 * active_entries
        ) / (torch.sum(active_entries)+1)
        
        return treatment_loss+outcome_loss, treatment_loss, outcome_loss


class CRNDataset(Dataset):

    def __init__(self, dataset_dict):
        super().__init__()
        self.data_dict = dataset_dict

    def __len__(self):
        return self.data_dict["current_covariates"].shape[0]

    def __getitem__(self, index):
        current_covariates = torch.from_numpy(
            self.data_dict["current_covariates"][index]
        ).float()
        # padding_tensor = torch.zeros_like(previous_treatments[0]).unsqueeze(0)
        # previous_treatments = torch.cat((padding_tensor, previous_treatments)).float()
        target_treatments = torch.from_numpy(
            self.data_dict["current_treatments"][index]
        ).float()
        target_outcomes = torch.from_numpy(self.data_dict["outputs"][index]).float()
        active_entries = torch.from_numpy(
            self.data_dict["active_entries"][index]
        ).float()
        if "init_states" in self.data_dict:
            if type(self.data_dict["init_states"][index]) == torch.Tensor:
                init_states = self.data_dict["init_states"][index]
            elif type(self.data_dict["init_states"][index]) == tuple:
                init_states = self.data_dict["init_states"][index]
            else:
                init_states = torch.from_numpy(
                    self.data_dict["init_states"][index]
                ).float()
            return (
                current_covariates,
                target_treatments,
                target_outcomes,
                active_entries,
                init_states
            )
        else:
            return (
                current_covariates,
                target_treatments,
                target_outcomes,
                active_entries
            )

