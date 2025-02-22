
import torch


class FlipGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.alpha
        return -grad_output * alpha, None



