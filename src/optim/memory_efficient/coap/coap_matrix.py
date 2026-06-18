# Copyright (C) 2025 ByteDance
# Licensed under the Apache License, Version 2.0

import torch
from .orthogonal import compute_orthogonal


class Projector:
    """Dummy projector — passes gradients through unchanged."""

    def project(self, full_rank_grad, state):
        return full_rank_grad

    def project_back(self, low_rank_grad):
        return low_rank_grad


class MatrixCOAP:
    """
    COAP projection for 2-D weight matrices.

    Args:
        rank: low-rank subspace dimension.
        update_interval: steps between cheap projection updates (eq. 5).
        reproject_factor: multiplier for full SVD recomputation period
            (full recompute every update_interval * reproject_factor steps).
        restore_state: whether to rotate Adam moments into the new basis.
        scale: scalar applied on project_back.
    """

    def __init__(self, rank, update_interval=64, reproject_factor=5,
                 restore_state=False, scale=1.0):
        self.rank = rank
        self.update_interval = update_interval
        self.reproject_factor = reproject_factor
        self.restore_state = restore_state
        self.scale = scale
        self.ortho_matrix = None

    @torch.no_grad()
    def update_proj(self, proj, proj_M, proj_type, grad, lr=0.1):
        """Gradient-descent update of the projection matrix."""
        proj.requires_grad = True
        grad.requires_grad = False
        proj_M.requires_grad = False

        with torch.enable_grad():
            if proj_type == "right":
                exp_avg = proj_M @ proj
                loss = torch.nn.functional.mse_loss(grad @ proj.T @ proj, grad)
            else:
                exp_avg = proj @ proj_M
                loss = torch.nn.functional.mse_loss(proj @ proj.T @ grad, grad)
            cosine_sim = torch.nn.functional.cosine_similarity(exp_avg, grad, dim=1)
            loss = loss * (1 - cosine_sim.mean())
            loss.backward()

        torch.nn.utils.clip_grad_norm_(proj, max_norm=1.0)
        proj.data.add_(proj.grad, alpha=-lr)
        proj.requires_grad = False
        if proj.grad is not None:
            proj.grad.zero_()
        return proj

    def restore(self, low_rank_grad):
        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            return torch.matmul(low_rank_grad, self.ortho_matrix)
        else:
            return torch.matmul(self.ortho_matrix, low_rank_grad)

    def project_back(self, low_rank_grad):
        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            full = torch.matmul(low_rank_grad, self.ortho_matrix)
        else:
            full = torch.matmul(self.ortho_matrix, low_rank_grad)
        return full * self.scale

    def get_orthogonal_matrix(self, weights, rank, type):
        matrix = weights.data
        float_data = matrix.dtype == torch.float
        original_device = matrix.device
        original_dtype = matrix.dtype
        if not float_data:
            matrix = matrix.float()

        omega = None
        if type == "right":
            if self.ortho_matrix is not None:
                omega = self.ortho_matrix.T
            Vh = compute_orthogonal(matrix, rank, omega)
            if not float_data:
                Vh = Vh.to(original_device).to(original_dtype)
            return Vh
        elif type == "left":
            if self.ortho_matrix is not None:
                omega = self.ortho_matrix
            Vh = compute_orthogonal(matrix.T, rank, omega)
            A = Vh.T
            if not float_data:
                A = A.to(original_device).to(original_dtype)
            return A
        else:
            raise ValueError(f"type must be 'left' or 'right', got {type!r}")

    def project(self, full_rank_grad, state):
        step = state["step"]
        _type = "right" if full_rank_grad.shape[0] >= full_rank_grad.shape[1] else "left"

        if self.ortho_matrix is None:
            self.ortho_matrix = self.get_orthogonal_matrix(
                full_rank_grad, self.rank, type=_type
            )
        elif step % self.update_interval == 0:
            exp_avg = state["exp_avg"]
            if self.restore_state:
                proj_exp_avg = self.restore(exp_avg)
                proj_exp_avg_sq = self.restore(state["exp_avg_sq"] ** 0.5)

            if step % (self.update_interval * self.reproject_factor) == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type=_type
                )
            else:
                self.ortho_matrix = self.update_proj(
                    self.ortho_matrix, exp_avg, _type, full_rank_grad
                )

            if self.restore_state:
                if _type == "right":
                    state["exp_avg"] = torch.matmul(proj_exp_avg, self.ortho_matrix.T)
                    state["exp_avg_sq"] = torch.matmul(proj_exp_avg_sq, self.ortho_matrix.T) ** 2
                else:
                    state["exp_avg"] = torch.matmul(self.ortho_matrix.T, proj_exp_avg)
                    state["exp_avg_sq"] = torch.matmul(self.ortho_matrix.T, proj_exp_avg_sq) ** 2

        if _type == "right":
            return torch.matmul(full_rank_grad, self.ortho_matrix.T)
        else:
            return torch.matmul(self.ortho_matrix.T, full_rank_grad)
