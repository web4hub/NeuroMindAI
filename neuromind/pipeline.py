"""High-level text inference pipeline."""
from __future__ import annotations

import torch

from .generation import generate


class NeuroMindPipeline:
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def __call__(self, prompt, max_new_tokens=32, temperature=1.0, top_k=0):
        ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        out = generate(self.model, ids, max_new_tokens, temperature, top_k)
        return self.tokenizer.decode(out[0].tolist())
