import json
import math
import os
import torch
from itertools import repeat
from loguru import logger


__all__ = ["MuonLite", "NuMuon"]


# Newton-Schulz polynomial coefficients for matrix sign approximation
_coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
# Numerical stability safety factor (exclude last polynomial)
_coeffs_list = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in _coeffs_list[:-1]
] + [_coeffs_list[-1]]


class OptimizedNewtonSchulz:
    """Newton-Schulz iteration with per-shape torch.compile caching."""

    def __init__(self):
        self.shape_cache = {}

    def _get_compiled_function(self, shape_key: tuple):
        @torch.compile(dynamic=False, fullgraph=True, backend="inductor")
        def compiled_func(G: torch.Tensor, steps: int) -> torch.Tensor:
            assert G.ndim >= 2
            X = G
            if G.size(-2) > G.size(-1):
                X = X.mT
            X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-8)

            hs = _coeffs_list[:steps] + list(repeat(_coeffs_list[-1], steps - len(_coeffs_list)))
            for a, b, c in hs:
                A = X @ X.mT
                B = b * A + c * A @ A
                X = a * X + B @ X

            if G.size(-2) > G.size(-1):
                X = X.mT
            return X

        return compiled_func

    def __call__(self, G: torch.Tensor, steps: int) -> torch.Tensor:
        shape_key = tuple(G.shape)
        if shape_key not in self.shape_cache:
            logger.info(f"Compiling Newton-Schulz for shape: {shape_key}")
            self.shape_cache[shape_key] = self._get_compiled_function(shape_key)
        return self.shape_cache[shape_key](G, steps)


zeropower_via_newtonschulz5 = OptimizedNewtonSchulz()


def _scheduled_rank_fraction(
    step: int,
    total_steps: int,
    *,
    start: float,
    end: float,
    warmup_fraction: float,
    decay_end_fraction: float,
    scheduler: str,
) -> float:
    """Rank fraction schedule used by NuMuon."""
    start = float(start)
    end = float(end)
    if scheduler == "fixed":
        return end

    if scheduler != "cosine":
        raise ValueError(f"Unknown NuMuon rank scheduler: {scheduler}")

    total_steps = max(1, int(total_steps))
    warmup_step = int(total_steps * warmup_fraction)
    decay_end_step = max(warmup_step + 1, int(total_steps * decay_end_fraction))

    if step < warmup_step:
        return start
    if step >= decay_end_step:
        return end

    progress = (step - warmup_step) / max(1, decay_end_step - warmup_step)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _rank_from_fraction(m: int, n: int, fraction: float) -> int:
    max_rank = min(m, n)
    if max_rank <= 1:
        return 1
    return min(max_rank, max(1, int(math.ceil(float(fraction) * max_rank))))


def _numuon_lmo_group_name(name: str) -> str:
    name = name.lower()
    if "gate_proj" in name:
        return "ffn_gate"
    if "up_proj" in name or "w12" in name:
        return "ffn_up"
    if "down_proj" in name or ("c_proj" in name and "mlp" in name):
        return "ffn_down"
    if "q_proj" in name:
        return "attn_q"
    if "k_proj" in name:
        return "attn_k"
    if "v_proj" in name:
        return "attn_v"
    if "o_proj" in name or ("c_proj" in name and "attn" in name):
        return "attn_o"
    return "other"


@torch.no_grad()
def _numuon_lmo_diagnostic(A: torch.Tensor, update: torch.Tensor, rank: int) -> dict:
    A32 = A.float()
    U32 = update.float()
    s_A = torch.linalg.svdvals(A32)
    s_U = torch.linalg.svdvals(U32)
    exact_obj = s_A[:rank].sum()
    approx_obj = (A32 * U32).sum()
    exact_obj_f = exact_obj.item()
    spectral_norm = s_U[0].item()
    nuclear_norm = s_U.sum().item()
    objective_ratio = approx_obj.item() / exact_obj_f if exact_obj_f != 0 else float("nan")
    return {
        "objective_ratio": objective_ratio,
        "approx_objective": approx_obj.item(),
        "exact_objective": exact_obj_f,
        "spectral_norm": spectral_norm,
        "spectral_violation": max(0.0, spectral_norm - 1.0),
        "nuclear_norm": nuclear_norm,
        "nuclear_budget": rank,
        "nuclear_violation": max(0.0, nuclear_norm - float(rank)),
        "fro_norm": U32.norm().item(),
        "fro_target": math.sqrt(rank),
        "effective_rank_1e_3": int((s_U > 1e-3).sum().item()),
    }


@torch.no_grad()
def block_krylov_topk_polar(
    A: torch.Tensor,
    rank: int,
    *,
    oversample: int = 8,
    krylov_iters: int = 2,
    warm_start: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate the rank-k polar factor U_k V_k^T with Block Krylov SVD.

    This follows the paper's randomized Block Krylov top-k SVD recipe: build a
    right Krylov subspace, compute a small SVD of A projected to that subspace,
    then discard singular values and keep the top singular directions.
    """
    if A.ndim != 2:
        raise ValueError(f"NuMuon expects a 2D matrix, got shape {tuple(A.shape)}")

    m, n = A.shape
    max_rank = min(m, n)
    rank = min(max_rank, max(1, int(rank)))
    block_size = min(max_rank, max(rank, rank + int(oversample)))
    krylov_iters = max(1, int(krylov_iters))

    A32 = A.float()
    if (
        warm_start is not None
        and warm_start.ndim == 2
        and warm_start.shape[0] == n
        and warm_start.numel() > 0
    ):
        B0 = warm_start[:, : min(block_size, warm_start.shape[1])].float()
        if B0.shape[1] < block_size:
            pad = torch.randn(
                n,
                block_size - B0.shape[1],
                device=A.device,
                dtype=torch.float32,
            )
            B0 = torch.cat([B0, pad], dim=1)
    else:
        B0 = torch.randn(n, block_size, device=A.device, dtype=torch.float32)

    B = torch.linalg.qr(B0, mode="reduced").Q
    basis_blocks = []
    for _ in range(krylov_iters):
        T = A32 @ B
        B = A32.mT @ T
        B = torch.linalg.qr(B, mode="reduced").Q
        basis_blocks.append(B)

    K = torch.cat(basis_blocks, dim=1)
    Q = torch.linalg.qr(K, mode="reduced").Q
    T = A32 @ Q
    U_t, _, Vh_t = torch.linalg.svd(T, full_matrices=False)
    U_k = U_t[:, :rank]
    V_k = Q @ Vh_t.mT[:, :rank]
    update = U_k @ V_k.mT
    return update.to(dtype=A.dtype), V_k.detach().to(dtype=A.dtype)


@torch.no_grad()
def mproj(m, msign_m, steps):
    """Top subspace projection: keep eigenval>1 subspace."""
    return (msign_m + zeropower_via_newtonschulz5(m - msign_m, steps)) / 2


@torch.no_grad()
def rank_v(tensor, top_ratio=1.0, lower_ratio=0.5):
    """Piecewise linear scaling for sharp subspace detection in AdamW params.

    Elements below lower_threshold → 0, above upper_threshold → 1,
    in between → linearly interpolated. Approximates a projection onto sharp subspace.
    """
    lower_threshold = lower_ratio * torch.mean(tensor)
    upper_threshold = top_ratio * torch.mean(tensor)
    range_size = upper_threshold - lower_threshold + 1e-12

    result = torch.zeros_like(tensor)
    mask_high = tensor >= upper_threshold
    result[mask_high] = 1.0

    mask_middle = (tensor >= lower_threshold) & (tensor < upper_threshold)
    if mask_middle.any():
        result[mask_middle] = (tensor[mask_middle] - lower_threshold) / range_size

    new_top_ratio = mask_high.sum() / tensor.numel()
    new_lower_ratio = torch.sum(tensor >= lower_threshold) / tensor.numel()
    return new_top_ratio, new_lower_ratio, result


def _cosine_lr_schedule(step, warmup_steps, total_steps, max_lr, min_ratio=0.1):
    """Cosine LR schedule with linear warmup (for internal lr_ratio computation)."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return max_lr * (min_ratio + (1 - min_ratio) * cosine_decay)


def _beta_scheduler(t, warmup, beta_final, beta_start, T_beta):
    """Schedule β₂ from β₁ (start) to β₂ (final) over warmup+T_flat period."""
    if T_beta > 0:
        if t >= T_beta:
            return beta_final
        elif t <= warmup:
            return beta_start
        else:
            return beta_start - (beta_start - beta_final) * (t - warmup) / (T_beta - warmup)
    return beta_final


# Parameter name patterns for block classification
# Reference block types: qk, vo, ffn, emb, out, norm
_QK_PATTERNS = ("c_attn", "q_proj", "k_proj")
_VO_PATTERNS = ("v_proj", "o_proj")
_FFN_PATTERNS = ("w12", "gate_proj", "up_proj", "down_proj")
_EMB_PATTERNS = ("wte", "embed")
_OUT_PATTERNS = ("lm_head", "head")
_NORM_PATTERNS = ("ln_", "norm")

# Block types that use Muon (2D), and their flat_mode
_MUON_BLOCKS = {"qk": 1, "vo": 1, "ffn": 1}
# Block types that use AdamW (1D/emb/out/norm)
_ADAMW_BLOCKS = {"emb": 2, "out": 2, "norm": 2}


def _classify_param(name):
    """Classify parameter into block type based on name patterns.

    Returns (block_type, flat_mode) where:
    - block_type: 'qk', 'vo', 'ffn', 'emb', 'out', 'norm', or None
    - flat_mode: 0=no schedule, 1=muon ramp, 2=adamw schedule
    """
    name_lower = name.lower()

    # QK attention (c_attn is combined QKV in llama.py — classify as qk)
    if any(p in name_lower for p in _QK_PATTERNS):
        return "qk", 1
    # VO attention
    if any(p in name_lower for p in _VO_PATTERNS):
        return "vo", 1
    # c_proj in attention context → vo (output projection)
    if "c_proj" in name_lower and ("attn" in name_lower or "attention" in name_lower):
        return "vo", 1

    if any(p in name_lower for p in _FFN_PATTERNS):
        return "ffn", 1
    # c_proj not in attn context → ffn (mlp down projection in llama.py)
    if "c_proj" in name_lower:
        return "ffn", 1

    if any(p in name_lower for p in _EMB_PATTERNS):
        return "emb", 2
    if any(p in name_lower for p in _OUT_PATTERNS):
        return "out", 2
    if any(p in name_lower for p in _NORM_PATTERNS):
        return "norm", 2

    return None, 0


class MuonLite(torch.optim.Optimizer):
    """MuonLite — Muon optimizer with LITE flat-direction acceleration.

    2D weight matrices (except embeddings/lm_head) use Muon + LITE.
    Embeddings and norm params use AdamW + LITE.
    lm_head uses AdamW without LITE.
    """

    def __init__(
        self,
        muon_params,
        adamw_params,
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        muon_theta: float = 0.95,
        ns_steps: int = 6,
        beta1: float = -0.25,
        beta2: float = 1.0,
        chi: float = 2.0,
        chi_adamw: float = 4.0,
        subspace_ratio: float = 0.1,
        lr_ratio_dict: dict = None,
        subspace_ratio_dict: dict = None,
        adamw_betas=(0.9, 0.95),
        adamw_eps: float = 1e-8,
        total_steps: int = 10000,
        warmup_steps: int = 1000,
        min_lr_ratio: float = 0.1,
        numuon: bool = False,
        numuon_fast: bool = True,
        numuon_rank_start: float = 1.0,
        numuon_rank_end: float = 0.25,
        numuon_rank_scheduler: str = "cosine",
        numuon_rank_warmup_fraction: float = 0.1,
        numuon_rank_decay_end_fraction: float = 1.0,
        numuon_krylov_iters: int = 2,
        numuon_oversample: int = 8,
        numuon_warm_start: bool = True,
        numuon_lmo_log_interval: int = 0,
        numuon_lmo_output_path: str = None,
        numuon_lmo_max_per_group: int = 1,
        numuon_lmo_groups: list[str] = None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            muon_theta=muon_theta,
            beta1=beta1,
            beta2=beta2,
            ns_steps=ns_steps,
            adamw_eps=adamw_eps,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )

        muon_params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params = [p for _, p in muon_params + adamw_params]
        super().__init__(params, defaults)

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.adamw_theta = adamw_betas[0]
        self.adamw_b2 = adamw_betas[1]
        self.smooth_ratio = 0.1
        # T_f: fraction of post-warmup period for flat ramp-up (cosine schedule)
        self.T_f = 0.5
        self.iter = 0
        self.numuon = bool(numuon)
        self.numuon_fast = bool(numuon_fast)
        self.numuon_rank_start = numuon_rank_start
        self.numuon_rank_end = numuon_rank_end
        self.numuon_rank_scheduler = numuon_rank_scheduler
        self.numuon_rank_warmup_fraction = numuon_rank_warmup_fraction
        self.numuon_rank_decay_end_fraction = numuon_rank_decay_end_fraction
        self.numuon_krylov_iters = numuon_krylov_iters
        self.numuon_oversample = numuon_oversample
        self.numuon_warm_start = numuon_warm_start
        self.numuon_lmo_log_interval = int(numuon_lmo_log_interval or 0)
        self.numuon_lmo_output_path = numuon_lmo_output_path
        self.numuon_lmo_max_per_group = max(1, int(numuon_lmo_max_per_group or 1))
        self.numuon_lmo_groups = set(numuon_lmo_groups or ("ffn_gate", "ffn_down", "ffn_up"))
        self.numuon_lmo_rank = int(os.environ.get("RANK", "0"))
        self._numuon_lmo_counts = {}
        self._numuon_lmo_iter = None

        # Build per-block dicts from simplified params (chi/chi_adamw/subspace_ratio),
        # then allow explicit per-block overrides via lr_ratio_dict/subspace_ratio_dict.
        # Block types: qk, vo, ffn (Muon), emb, out, norm (AdamW)
        lr_dict = dict(qk=chi, vo=chi, ffn=chi, emb=chi_adamw, out=1.0, norm=chi_adamw)
        sub_dict = dict(qk=subspace_ratio, vo=subspace_ratio, ffn=subspace_ratio,
                        emb=subspace_ratio, out=1.0, norm=subspace_ratio)
        if lr_ratio_dict is not None:
            lr_dict.update(lr_ratio_dict)
        if subspace_ratio_dict is not None:
            sub_dict.update(subspace_ratio_dict)

        # Classify each parameter into block type and assign LITE settings
        for name, p in muon_params:
            assert p.ndim == 2, f"Muon only supports 2D parameters, got {p.ndim}D for {name}"
            block_type, flat_mode = _classify_param(name)

            if block_type in _MUON_BLOCKS and sub_dict.get(block_type, 0) > 0:
                # Muon + LITE
                self.state[p]["use_muon"] = 1
                self.state[p]["subspace_ratio"] = sub_dict[block_type]
                self.state[p]["lr_ratio"] = lr_dict[block_type]
                self.state[p]["flat_warmup"] = flat_mode
            else:
                # Vanilla Muon (unrecognized block or LITE disabled for this block)
                self.state[p]["use_muon"] = 2
                self.state[p]["subspace_ratio"] = 1.0
                self.state[p]["lr_ratio"] = 1.0
                self.state[p]["flat_warmup"] = 0

            self.state[p]["name"] = name

        for name, p in adamw_params:
            block_type, flat_mode = _classify_param(name)
            self.state[p]["use_muon"] = 0
            self.state[p]["name"] = name

            if block_type in _ADAMW_BLOCKS:
                self.state[p]["subspace_ratio"] = sub_dict.get(block_type, 1.0)
                self.state[p]["lr_ratio"] = lr_dict.get(block_type, 1.0)
                self.state[p]["flat_warmup"] = flat_mode
            else:
                # Fallback AdamW (no LITE)
                self.state[p]["subspace_ratio"] = 1.0
                self.state[p]["lr_ratio"] = 1.0
                self.state[p]["flat_warmup"] = 0

    def get_lr_ratio(self, lr_ratio, flat_mode):
        """Compute time-varying LR amplification factor for flat directions."""
        if flat_mode == 0 or flat_mode is None:
            return lr_ratio

        if flat_mode == 2:
            # AdamW blocks: full chi at peak, decays to 1.0
            lr_f = _cosine_lr_schedule(
                self.iter, self.warmup_steps, self.total_steps,
                lr_ratio, self.min_lr_ratio / lr_ratio,
            )
            lr_s = _cosine_lr_schedule(
                self.iter, self.warmup_steps, self.total_steps,
                1.0, self.min_lr_ratio,
            ) + 1e-9
            return lr_f / lr_s

        if flat_mode == 1:
            # Muon blocks: 1.0 during warmup, gradual ramp-up, then decay
            if self.iter <= self.warmup_steps:
                return 1.0
            lr_f = _cosine_lr_schedule(
                self.iter, self.warmup_steps, self.total_steps,
                lr_ratio, self.min_lr_ratio / lr_ratio,
            )
            lr_s = _cosine_lr_schedule(
                self.iter, self.warmup_steps, self.total_steps,
                1.0, self.min_lr_ratio,
            ) + 1e-9
            lr_ratio_t = lr_f / lr_s
            # Gradual ramp with T_f
            lr_ratio_t = min(
                lr_ratio_t,
                1 + (self.iter - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                * (lr_ratio - 1) / self.T_f,
            )
            return lr_ratio_t

        return lr_ratio

    def get_numuon_rank(self, m, n):
        rank_fraction = _scheduled_rank_fraction(
            self.iter,
            self.total_steps,
            start=self.numuon_rank_start,
            end=self.numuon_rank_end,
            warmup_fraction=self.numuon_rank_warmup_fraction,
            decay_end_fraction=self.numuon_rank_decay_end_fraction,
            scheduler=self.numuon_rank_scheduler,
        )
        return _rank_from_fraction(m, n, rank_fraction), rank_fraction

    def _should_log_numuon_lmo(self, param_name):
        if not self.numuon or self.numuon_lmo_log_interval <= 0:
            return False, None, None
        if self.numuon_lmo_rank != 0 or not self.numuon_lmo_output_path:
            return False, None, None

        step_index = self.iter + 1
        if (
            step_index != 1
            and step_index != self.total_steps
            and step_index % self.numuon_lmo_log_interval != 0
        ):
            return False, None, None

        if self._numuon_lmo_iter != step_index:
            self._numuon_lmo_iter = step_index
            self._numuon_lmo_counts = {}

        group = _numuon_lmo_group_name(param_name)
        if group not in self.numuon_lmo_groups:
            return False, None, None

        count = self._numuon_lmo_counts.get(group, 0)
        if count >= self.numuon_lmo_max_per_group:
            return False, None, None
        self._numuon_lmo_counts[group] = count + 1
        return True, step_index, group

    def _write_numuon_lmo_record(self, record):
        os.makedirs(os.path.dirname(self.numuon_lmo_output_path), exist_ok=True)
        with open(self.numuon_lmo_output_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            muon_theta = group["muon_theta"]
            ns_steps = group["ns_steps"]

            # Schedule β₂ from β₁ to β₂_final
            T_beta = int(self.T_f * (self.total_steps - self.warmup_steps)) + self.warmup_steps
            beta2 = _beta_scheduler(
                self.iter, self.warmup_steps, group["beta2"], group["beta1"], T_beta,
            )

            # ── Muon + LITE params (2D, attn/ffn) ──
            for p in group["params"]:
                if self.state[p]["use_muon"] != 1:
                    continue
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                m, n = g.size(0), g.size(1)

                if "step" not in state:
                    state["step"] = 0
                    state["subspace_threshold_ratio"] = 1.0 / math.sqrt(min(m, n))
                    state["momentum"] = torch.zeros_like(g)

                k = int(state["subspace_ratio"] * min(m, n)) + 1
                k = min(k, min(m, n) - 1)

                lr_times = self.get_lr_ratio(state["lr_ratio"], state["flat_warmup"])

                # Nesterov momentum
                state["momentum"] = state["momentum"] * muon_theta + g * (1 - muon_theta)
                M = state["momentum"] + g * (1 - muon_theta) / muon_theta

                m_ns = zeropower_via_newtonschulz5(M, ns_steps)

                # Sharp subspace projection via composite NS
                thres_r = M.norm() * state["subspace_threshold_ratio"] + 1e-9
                state_P = mproj(M / thres_r, m_ns, ns_steps)

                # Dynamic threshold adjustment
                if state_P.norm() > math.sqrt(k):
                    state["subspace_threshold_ratio"] *= 1.05
                else:
                    state["subspace_threshold_ratio"] *= 0.95
                state["subspace_threshold_ratio"] = min(1.0, state["subspace_threshold_ratio"])

                # Flat complement projector
                P_flat_smooth = torch.eye(n, dtype=state_P.dtype, device=state_P.device) - state_P.t() @ state_P

                # Hessian damping
                hessian_damping = zeropower_via_newtonschulz5(g, ns_steps)
                hessian_damping_flat = hessian_damping @ P_flat_smooth

                # Update: NS(M̃) + β₁*NS(g) + (β₂-β₁)*NS(g)@Q_flat
                update = m_ns + group["beta1"] * hessian_damping + (beta2 - group["beta1"]) * hessian_damping_flat
                # Amplify LR in flat directions
                update = update + (lr_times - 1) * update @ P_flat_smooth

                # Extra weight decay in flat directions
                flat_data_wd = p.data @ P_flat_smooth
                wd2 = (lr_times - 1) * wd

                if state["step"] == 0:
                    logger.debug(
                        f"{state['name']}, muon+lite, beta1={group['beta1']}, beta2={group['beta2']}, "
                        f"subspace={state['subspace_ratio']}, lr_ratio={state['lr_ratio']}"
                    )

                state["step"] += 1
                p.data.mul_(1 - lr * wd)
                p.data.add_(flat_data_wd, alpha=-lr * wd2)
                p.data.add_(update, alpha=-0.2 * lr * math.sqrt(max(m, n)))

            # ── Vanilla Muon params (2D, no LITE) ──
            numuon_fast_active = self.numuon and self.numuon_fast
            if numuon_fast_active:
                from .numuon_fast import numuon_fast_step
                numuon_fast_step(self, group, lr, wd, muon_theta)
            for p in group["params"]:
                if numuon_fast_active or self.state[p]["use_muon"] != 2:
                    continue
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                m, n = g.size(0), g.size(1)

                if "step" not in state:
                    state["step"] = 0

                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(g)

                state["momentum"] = state["momentum"] * muon_theta + g * (1 - muon_theta)
                M = state["momentum"] + g * (1 - muon_theta) / muon_theta

                if state["step"] == 0:
                    if self.numuon:
                        logger.debug(
                            f"{state['name']}, numuon, scheduler={self.numuon_rank_scheduler}, "
                            f"rank_start={self.numuon_rank_start}, rank_end={self.numuon_rank_end}, "
                            f"krylov_iters={self.numuon_krylov_iters}, oversample={self.numuon_oversample}"
                        )
                    else:
                        logger.debug(f"{state['name']}, vanilla_muon")

                if self.numuon:
                    rank, rank_fraction = self.get_numuon_rank(m, n)
                    warm_start = state.get("numuon_basis") if self.numuon_warm_start else None
                    u, basis = block_krylov_topk_polar(
                        M,
                        rank,
                        oversample=self.numuon_oversample,
                        krylov_iters=self.numuon_krylov_iters,
                        warm_start=warm_start,
                    )
                    if self.numuon_warm_start:
                        state["numuon_basis"] = basis
                    state["numuon_rank"] = rank
                    state["numuon_rank_fraction"] = rank_fraction
                    should_log_lmo, lmo_step, lmo_group = self._should_log_numuon_lmo(state["name"])
                    if should_log_lmo:
                        record = {
                            "iter": lmo_step,
                            "param": state["name"],
                            "group": lmo_group,
                            "shape": [m, n],
                            "rank": rank,
                            "rank_fraction": rank_fraction,
                            "krylov_iters": self.numuon_krylov_iters,
                            "oversample": self.numuon_oversample,
                            "warm_start": bool(warm_start is not None),
                        }
                        record.update(_numuon_lmo_diagnostic(M, u, rank))
                        self._write_numuon_lmo_record(record)
                        logger.info(
                            "NuMuonLMO: iter={} group={} param={} ratio={:.4f} spec={:.6f} nuc={:.3f}/{}",
                            lmo_step,
                            lmo_group,
                            state["name"],
                            record["objective_ratio"],
                            record["spectral_norm"],
                            record["nuclear_norm"],
                            rank,
                        )
                else:
                    u = zeropower_via_newtonschulz5(M, ns_steps)

                state["step"] += 1
                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-0.2 * lr * math.sqrt(max(m, n)))

            # ── AdamW params (emb, out, norm, fallback) ──
            eps = group["adamw_eps"]
            adam_b2 = self.adamw_b2
            adam_theta = self.adamw_theta

            for p in group["params"]:
                if self.state[p]["use_muon"] != 0:
                    continue
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                    state["upper_topk_ratio"] = 1.0
                    state["lower_topk_ratio"] = 0.5

                if state["step"] == 0:
                    logger.debug(
                        f"{state['name']}, adamw, subspace={state['subspace_ratio']}, "
                        f"lr_ratio={state['lr_ratio']}"
                    )

                state["moment1"] = adam_theta * state["moment1"] + (1 - adam_theta) * g
                state["moment2"] = state["moment2"] * adam_b2 + (g ** 2) * (1 - adam_b2)

                lr_times = self.get_lr_ratio(state["lr_ratio"], state["flat_warmup"])

                smooth_ratio = min(1.0 - state["subspace_ratio"], self.smooth_ratio)

                new_upper, new_lower, state_p = rank_v(
                    state["moment2"], state["upper_topk_ratio"], state["lower_topk_ratio"],
                )

                # Dynamic threshold adjustment for AdamW subspace detection
                if new_lower > smooth_ratio + state["subspace_ratio"]:
                    state["lower_topk_ratio"] *= 1.05
                else:
                    state["lower_topk_ratio"] *= 0.95
                if new_upper > state["subspace_ratio"]:
                    state["upper_topk_ratio"] *= 1.05
                else:
                    state["upper_topk_ratio"] *= 0.95
                state["lower_topk_ratio"] = min(
                    state["lower_topk_ratio"], state["upper_topk_ratio"] * 0.95,
                )

                # Edge cases: all sharp or all flat
                if state["subspace_ratio"] < 1e-5:
                    state_p = torch.zeros_like(g)
                elif state["subspace_ratio"] > 1.0 - 1e-5:
                    state_p = torch.ones_like(g)

                update = state["moment1"] / (state["moment2"].sqrt() + eps)
                bias_correction1 = 1 - adam_theta ** (state["step"] + 1)
                bias_correction2 = 1 - adam_b2 ** (state["step"] + 1)
                scale = bias_correction1 / bias_correction2 ** 0.5

                # Amplify flat directions
                update = update + (lr_times - 1) * (1 - state_p) * update

                # Extra weight decay in flat directions
                flat_data_wd = (1 - state_p) * p.data
                wd2 = (lr_times - 1) * wd
                p.data.mul_(1 - lr * wd)
                p.data.add_(flat_data_wd, alpha=-lr * wd2)

                p.data.add_(update, alpha=-lr / scale)
                state["step"] += 1

        self.iter += 1
        return loss


class NuMuon(MuonLite):
    """NuMuon wrapper using MuonLite's parameter handling with NuMuon updates."""

    def __init__(self, *args, **kwargs):
        kwargs["numuon"] = True
        kwargs["beta1"] = 0.0
        kwargs["beta2"] = 0.0
        kwargs["chi"] = 1.0
        kwargs["chi_adamw"] = 1.0
        kwargs["subspace_ratio"] = 0.0
        super().__init__(*args, **kwargs)
