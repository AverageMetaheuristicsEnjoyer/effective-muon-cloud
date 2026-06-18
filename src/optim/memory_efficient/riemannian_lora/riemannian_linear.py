import math
import torch
import torch.nn as nn


class _WeightHolder(nn.Module):
    """Minimal module that holds a single 'weight' nn.Parameter.

    Named sub-modules like `lora_A` and `lora_B` of this class expose a
    `.weight` attribute so the Riemannian optimizer can look them up by the
    conventional name 'lora_A.weight' / 'lora_B.weight'.
    """

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight)


class RiemannianLoraLinear(nn.Module):
    """Linear layer whose weight is parametrized on the Stiefel manifold.

    The effective weight is W_approx = A_active @ B_active, where:
        A_active = lora_A.weight.T[:, :rank]   shape (out, rank)  -- Stiefel factor
        B_active = lora_B.weight.T[:rank, :]   shape (rank, in)   -- scale factor

    Stored parameter shapes (as expected by RiemannianLora.step()):
        lora_A.weight : (2*rank, out_features)
        lora_B.weight : (in_features, 2*rank)

    Forward: y = x @ B_active.T @ A_active.T  -- equivalent to x @ W_approx.T
    """

    def __init__(self, linear: nn.Linear, rank: int, init: str = "orth"):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank

        dtype = linear.weight.dtype
        device = linear.weight.device

        # ── lora_A.weight: (2r, out_features) ───────────────────────────────
        # First r rows form an orthonormal system (A_active on Stiefel).
        # Last r rows are zero (extended for the Riemannian parametrization).
        A_init = torch.zeros(2 * rank, self.out_features, dtype=dtype, device=device)
        Q_A = torch.linalg.qr(
            torch.randn(self.out_features, rank, dtype=dtype, device=device)
        ).Q  # (out, r) orthonormal
        A_init[:rank, :] = Q_A.T  # store as (r, out) in the first r rows

        # ── lora_B.weight: (in_features, 2r) ────────────────────────────────
        # First r columns hold B_active, last r columns are an orthonormal
        # complement (maintained by the retraction).
        B_init = torch.zeros(self.in_features, 2 * rank, dtype=dtype, device=device)
        if init == "orth":
            # Scale B so ||W_approx||_F ≈ std of original linear init
            target_std = math.sqrt(2.0 / self.in_features)  # Kaiming
            # ||A_active @ B_active||_F ≈ sqrt(rank) * ||B_active||_F / sqrt(out)
            b_std = target_std * math.sqrt(self.out_features / rank)
            B_init[:, :rank] = torch.randn(
                self.in_features, rank, dtype=dtype, device=device
            ) * b_std
        else:
            # "zero" init: model starts from zero weights (adapter style)
            pass  # B_init stays zero

        self.lora_A = _WeightHolder(A_init)
        self.lora_B = _WeightHolder(B_init)

        # Keep bias as a plain parameter updated by the regular (AdamW) optimizer.
        self.bias = (
            nn.Parameter(linear.bias.data.clone())
            if linear.bias is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.rank
        # A_active: (out, r)
        A_active = self.lora_A.weight.T[:, :r]
        # B_active: (r, in)
        B_active = self.lora_B.weight.T[:r, :]
        # y = x (*, in) @ B_active.T (in, r) @ A_active.T (r, out) = (*, out)
        out = x @ B_active.T @ A_active.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}"
        )
