from torch import nn
from typing import Optional
import torch

class LipLinear(nn.Module):
    # if lip_const is None, then this is equal to nn.Linear
    def __init__(self, input_dim:int, output_dim:int, lip_const: Optional[float]) -> None:
        super(LipLinear, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

        self.lip_const = lip_const
    
    def forward(self, x):
        out = self.linear(x)
        if self.lip_const is not None:
            # warning: this may be wrong
            mat_norm = torch.linalg.matrix_norm(self.linear.weight, ord=2)
            # warning: this may be wrong
            out = (self.lip_const/(mat_norm+1e-6))*out
        return out

class LipDecoder(nn.Module):

    
    # warning: this module is batch first by default
    # warning: lip_const must be None or bigger than one
    # this module adapted from coupled forgetting lstm, to better ensure Lipschitz continuous
    def __init__(self, input_dim: int, hidden_dim: int, lip_const: float) -> None:        
        super(LipDecoder, self).__init__()

        self.hidden_dim = hidden_dim

        self.linear = LipLinear(input_dim+hidden_dim, hidden_dim, lip_const)
    
    # warning: x should be batch first
    def forward(self, x:torch.Tensor, hx: Optional[tuple[torch.Tensor, torch.Tensor]]=None):
        num_batch = x.shape[0]
        num_sequence = x.shape[1]
        if hx is None:
            hd = torch.zeros(num_batch, self.hidden_dim,
                             dtype=x.dtype, device=x.device)
            cell = torch.zeros_like(hd)
        else:
            hd, cell = hx
            hd = hd[0]
            cell = cell[0]
        
        output = torch.zeros(num_batch, num_sequence, self.hidden_dim,
                             dtype=x.dtype, device=x.device)

        for i in range(num_sequence):
            input = torch.concatenate([x[:,i,:],hd], dim=1)
            o = nn.functional.tanh(self.linear(input))
            output[:,i,:] = o
        return output, (hd.reshape((1,)+hd.shape), cell.reshape((1,)+cell.shape))
        


