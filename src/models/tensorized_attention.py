"""Block-Term-Decomposition attention from arXiv:1906.09777.

The paper calls this operation *Multi-linear Attention*.  A single set of
Q/K/V projections is shared by several diagonal Tucker cores.  Two output
forms are provided:

``split_concat``
    The paper/author-code form.  It builds the third-order token tensor,
    flattens every pair of token axes, and applies :math:`W^O`.  This is the
    closest implementation of Eq. (8), but has cubic sequence complexity and
    is intended for the short (30--100 token) contexts used in the paper.

``reconstruction``
    The Corollary-1 form.  It sums one tensor axis before :math:`W^O`, reducing
    memory from O(T^3) to O(T^2).  This is useful for longer autoregressive
    language-model experiments while retaining the shared Q/K/V projections
    and learned diagonal BTD core.

The reference implementation published with the paper ignores its attention
mask.  Here causality is enabled by default so next-token training cannot look
at future tokens.  It can be disabled explicitly for reproduction studies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TensorizedAttention(nn.Module):
    """Multi-linear attention with shared projections and diagonal BTD cores."""

    MODES = ("split_concat", "reconstruction")

    def __init__(
        self,
        d_model: int,
        rank: int,
        num_cores: int,
        max_sequence_length: int,
        *,
        mode: str = "reconstruction",
        causal: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        query_chunk_size: int = 8,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if rank <= 0:
            raise ValueError("tensorized rank must be positive")
        if num_cores <= 0:
            raise ValueError("tensorized num_cores must be positive")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if mode not in self.MODES:
            raise ValueError(f"Unknown tensorized attention mode {mode!r}; expected {self.MODES}")
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")

        self.d_model = d_model
        self.rank = rank
        self.num_cores = num_cores
        self.max_sequence_length = max_sequence_length
        self.mode = mode
        self.causal = causal
        self.query_chunk_size = query_chunk_size

        # Wq, Wk and Wv are deliberately shared across all BTD core terms.
        self.q_proj = nn.Linear(d_model, rank, bias=bias)
        self.k_proj = nn.Linear(d_model, rank, bias=bias)
        self.v_proj = nn.Linear(d_model, rank, bias=bias)

        # Eq. (6): only the diagonal of every core is trainable.  Store that
        # diagonal directly instead of materialising a rank^3 mostly-zero tensor.
        self.core_logits = nn.Parameter(torch.empty(num_cores, rank))
        nn.init.uniform_(self.core_logits, 0.0, 1.0)

        output_features = (
            max_sequence_length * max_sequence_length
            if mode == "split_concat"
            else max_sequence_length
        )
        self.o_proj = nn.Linear(output_features, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def core_probabilities(self) -> torch.Tensor:
        """Return the softmax-normalised diagonal entries of every core."""
        return F.softmax(self.core_logits, dim=-1)

    def _mean_core(self, dtype: torch.dtype) -> torch.Tensor:
        # Eq. (8) averages h block terms.  Since their factor matrices are
        # shared, averaging the diagonal vectors is algebraically identical and
        # avoids constructing one cubic tensor per core.
        return self.core_probabilities().mean(dim=0).to(dtype=dtype)

    def _validate_input(self, x: torch.Tensor) -> tuple[int, int, int]:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, sequence, hidden] input, got {tuple(x.shape)}")
        batch, sequence, hidden = x.shape
        if hidden != self.d_model:
            raise ValueError(f"Expected hidden size {self.d_model}, got {hidden}")
        if sequence > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence} exceeds configured maximum "
                f"{self.max_sequence_length}"
            )
        return batch, sequence, hidden

    def _split_concat(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        core: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. (8): split/concat the third-order tensor, then apply W^O."""
        _, sequence, _ = q.shape
        device = q.device

        token_positions = torch.arange(sequence, device=device)
        flattened_positions = (
            token_positions[:, None] * self.max_sequence_length
            + token_positions[None, :]
        ).reshape(-1)

        outputs = []
        for start in range(0, sequence, self.query_chunk_size):
            end = min(start + self.query_chunk_size, sequence)
            # A[b,i,j,k] = sum_r g[r] Q[b,i,r] K[b,j,r] V[b,k,r].
            tensor_block = torch.einsum(
                "r,bir,bjr,bkr->bijk", core, q[:, start:end], k, v
            )

            if self.causal:
                query_positions = token_positions[start:end]
                key_visible = token_positions[None, :] <= query_positions[:, None]
                pair_visible = key_visible[:, :, None] & key_visible[:, None, :]
                tensor_block = tensor_block.masked_fill(
                    ~pair_visible.unsqueeze(0), 0.0
                )

            flattened = tensor_block.reshape(q.size(0), end - start, -1)
            if sequence == self.max_sequence_length:
                projection_input = flattened
            else:
                # Calling the projection module (rather than slicing .weight)
                # keeps this path compatible with TuckerLinear.
                projection_input = flattened.new_zeros(
                    flattened.shape[0],
                    flattened.shape[1],
                    self.max_sequence_length * self.max_sequence_length,
                )
                projection_input.index_copy_(
                    -1, flattened_positions, flattened
                )
            outputs.append(self.o_proj(projection_input))

        return torch.cat(outputs, dim=1)

    def _reconstruction(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        core: torch.Tensor,
    ) -> torch.Tensor:
        """Corollary-1 reconstruction with an optional causal prefix sum."""
        _, sequence, _ = q.shape
        if self.causal:
            # Sum the key axis j only over the visible prefix for each query i.
            key_sum = k.cumsum(dim=1)
        else:
            key_sum = k.sum(dim=1, keepdim=True).expand(-1, sequence, -1)

        query_factors = q * key_sum * core.view(1, 1, -1)
        reconstructed = torch.einsum("bir,bkr->bik", query_factors, v)

        if self.causal:
            causal_mask = torch.ones(
                sequence, sequence, dtype=torch.bool, device=q.device
            ).tril()
            reconstructed = reconstructed.masked_fill(~causal_mask.unsqueeze(0), 0.0)

        if sequence < self.max_sequence_length:
            reconstructed = F.pad(
                reconstructed,
                (0, self.max_sequence_length - sequence),
            )
        return self.o_proj(reconstructed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        core = self._mean_core(q.dtype)

        if self.mode == "split_concat":
            output = self._split_concat(q, k, v, core)
        else:
            output = self._reconstruction(q, k, v, core)
        return self.dropout(output)

    @property
    def attention_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, rank={self.rank}, num_cores={self.num_cores}, "
            f"max_sequence_length={self.max_sequence_length}, mode={self.mode!r}, "
            f"causal={self.causal}, query_chunk_size={self.query_chunk_size}"
        )


def tensorized_attention_from_config(config) -> TensorizedAttention:
    """Construct :class:`TensorizedAttention` from the project's CLI config."""
    return TensorizedAttention(
        d_model=config.n_embd,
        rank=config.tensorized_rank,
        num_cores=config.tensorized_num_cores,
        max_sequence_length=config.sequence_length,
        mode=config.tensorized_mode,
        causal=config.tensorized_causal,
        dropout=config.dropout,
        bias=config.bias,
        query_chunk_size=config.tensorized_query_chunk_size,
    )
