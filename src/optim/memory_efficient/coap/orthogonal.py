# Copyright (C) 2025 ByteDance
# Licensed under the Apache License, Version 2.0

import torch


@torch.no_grad()
def compute_orthogonal(in_matrix, rank, omega=None):
    """
    Randomized SVD-based orthogonal basis computation.

    Args:
        in_matrix: [m, n]
        rank: target rank
        omega: optional warm-start matrix [n, rank]

    Returns:
        Vh: [rank, n]
    """
    device = in_matrix.device
    m, n = in_matrix.shape

    if omega is None:
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            generator = torch.Generator(device=device).manual_seed(3407)
            omega = torch.randn([n, rank], device=device, generator=generator)
    if omega.data.dtype != torch.float:
        omega = omega.data.float().to(device)

    Y = in_matrix @ omega
    Q, _ = torch.linalg.qr(Y, "reduced")  # [m, r]
    B = Q.T @ in_matrix  # [r, n]
    _, _, Vh = torch.linalg.svd(B, full_matrices=False)
    return Vh
