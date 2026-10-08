"""Optimizer and scheduler construction."""
from __future__ import annotations
import torch

def build_optimizer(model,lr=3e-4,weight_decay=0.1):
    return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
