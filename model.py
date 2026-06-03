"""
model.py — Deep Averaging Network (DAN) classifier.

Architecture (trained from scratch, no pretrained weights, no CNN):
  1. Embedding layer: each token gets a learned vector of size embed_dim
  2. Mean-pool: average all (non-padding) token embeddings into one vector
  3. MLP: two linear layers with ReLU → 2-class output (real vs fake)

The key hook for suppression: before mean-pooling, multiply each token
embedding by suppress_mask[token_id]. Setting that weight to 0 makes the
token invisible to the mean-pool, so the model cannot use it at all.
"""

import torch
import torch.nn as nn


class DeepAveragingNetwork(nn.Module):
    """
    DAN: the simplest possible neural text classifier.
    Deliberately simple — the point is that even this model benefits from
    spurious-token suppression, showing the method is architecture-agnostic.
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128,
                 num_classes=2, pad_idx=0):
        super().__init__()
        # One learned vector per vocabulary entry.
        # padding_idx=pad_idx → padding positions always produce a zero vector.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_ids, padding_mask=None, suppress_mask=None):
        """
        Args:
            token_ids     (batch, seq_len)  — integer word indices
            padding_mask  (batch, seq_len)  — 1.0 for real tokens, 0.0 for padding
            suppress_mask (vocab_size,)     — 1.0 for normal tokens, 0.0 for spurious

        suppress_mask is the entropy-debiasing mechanism:
          suppress_mask[token_ids] looks up each token's suppression weight.
          Multiplying embeddings by 0 zeros them out → invisible to mean-pool.
          The model gradient cannot flow back through a zeroed embedding during
          training, so it never learns weights for spurious tokens.
        """
        embeds = self.embedding(token_ids)      # → (batch, seq_len, embed_dim)

        # Step 1: apply suppression weights (makes spurious tokens invisible)
        if suppress_mask is not None:
            # Index suppress_mask by token_ids → shape (batch, seq_len)
            # Unsqueeze to broadcast across embed_dim dimension
            weights = suppress_mask[token_ids].unsqueeze(-1)   # (batch, seq_len, 1)
            embeds  = embeds * weights

        # Step 2: mean-pool over non-padding positions only
        if padding_mask is not None:
            embeds  = embeds * padding_mask.unsqueeze(-1)
            lengths = padding_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled  = embeds.sum(dim=1) / lengths              # (batch, embed_dim)
        else:
            pooled = embeds.mean(dim=1)

        # Step 3: classify
        return self.mlp(pooled)                                 # (batch, 2)
