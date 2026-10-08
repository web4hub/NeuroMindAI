from __future__ import annotations

import torch


class NeuroMindPipeline:
    """High-level text-generation pipeline for NeuroMindAI."""

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer

        if device is None:
            device = next(model.parameters()).device

        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

        if temperature < 0:
            raise ValueError("temperature must be >= 0")

        if top_k < 0:
            raise ValueError("top_k must be >= 0")

        token_ids = self.tokenizer.encode(prompt)

        if not token_ids:
            raise ValueError("tokenizer produced an empty sequence")

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        return self.tokenizer.decode(output_ids[0].tolist())
