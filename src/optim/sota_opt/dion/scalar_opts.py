import torch
from torch import Tensor
from typing import Generator, List
import math


@torch.compile(fullgraph=True)
def adamw_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    V: Tensor,  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
):
    """
    AdamW optimizer algorithm.
    """
    assert X.shape == G.shape
    assert X.shape == M.shape

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    M.lerp_(G.to(M.dtype), 1 - beta1)
    # V = beta2 * V + (1 - beta2) * G * G
    V.mul_(beta2).addcmul_(G, G, value=1 - beta2)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = V.sqrt().div_(bias_correction2_sqrt).add_(epsilon)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1

    # Apply weight decay
    X.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - adj_lr * M / denom
    X.addcdiv_(M, denom, value=-adj_lr)


@torch.compile(fullgraph=True)
def lion_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
):
    """
    Lion optimizer algorithm. Sign update should guarantee RMS norm equal to 1.
    """
    assert X.shape == G.shape
    assert X.shape == M.shape

    G = G.to(M.dtype)

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * G)
    U = M.lerp(G, 1 - beta1).sign_()

    # Update momentum with new gradient
    # M = beta2 * M + (1 - beta2) * G
    M.lerp_(G, 1 - beta2)

    # Apply weight decay
    X.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - lr * U
    X.add_(U, alpha=-lr)

@torch.compile(fullgraph=True)
def adan_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # First moment buffer (m_t) (modified in place)
    V: Tensor,  # Second moment buffer (n_t) (modified in place)
    D: Tensor,  # Difference moment buffer (v_t) (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    beta3: Tensor,  # Beta 3 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
):
    """
    AdaN optimizer algorithm.
    """
    assert X.shape == G.shape
    assert X.shape == M.shape
    assert X.shape == V.shape
    assert X.shape == D.shape

    lr = float(lr)
    beta1 = float(beta1)
    beta2 = float(beta2)
    beta3 = float(beta3)
    weight_decay = float(weight_decay)

    G = G.to(M.dtype)

    # m_k = (1 - beta1) * m_{k-1} + beta1 * g_k
    M.lerp_(G, 1 - beta1)

    # v_k = (1 - beta2) * v_{k-1} + beta2 * (g_k - g_{k-1})
    # D = g_k - g_{k-1}
    D.sub_(G).neg_()  # D = -(D - G) = G - D
    # v_k = (1 - beta2) * v_{k-1} + beta2 * D
    V.lerp_(D, 1 - beta2)

    # n_k = (1 - beta3) * n_{k-1} + beta3 * [g_k + (1 - beta2) * (g_k - g_{k-1})]^2
    # g_k + (1 - beta2) * (g_k - g_{k-1}) = g_k + (1 - beta2) * D
    D.mul_(1 - beta2).add_(G)  # D now holds [g_k + (1 - beta2) * (g_k - g_{k-1})]
    # n_k = (1 - beta3) * n_{k-1} + beta3 * D^2
    V.mul_(beta3).addcmul_(D, D, value=1 - beta3)

    # Bias correction
    bias_correction1 = 1 - math.pow(beta1, step)
    bias_correction2 = 1 - math.pow(beta2, step)
    bias_correction3_sqrt = math.sqrt(1 - math.pow(beta3, step))

    # eta_k = lr / (sqrt(n_k) + epsilon)
    denom = V.sqrt().div_(bias_correction3_sqrt).add_(epsilon)

    # Compute step sizes
    step_size = lr / bias_correction1
    step_size_diff = lr * beta2 / bias_correction2

    # X = X / (1 + lr * weight_decay)
    X.div_(1 + lr * weight_decay)

    # X = X - step_size * m_k / denom - step_size_diff * v_k / denom
    X.addcdiv_(M, denom, value=-step_size)
    X.addcdiv_(V, denom, value=-step_size_diff)

    D.copy_(G)

@torch.compile(fullgraph=True)
def adamw_update_foreach(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
):
    """
    AdamW optimizer algorithm (foreach implementation).
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)
    assert batch_size == len(V)

    ### ========= ###
    device = X[0].device
    lr = lr.to(device)
    beta1 = beta1.to(device)
    beta2 = beta2.to(device)
    weight_decay = weight_decay.to(device)
    ### ========== ###

    M_dtype = M[0].dtype
    V_dtype = V[0].dtype

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    G = [g.to(dtype=M_dtype) for g in G]
    torch._foreach_lerp_(M, G, [1 - beta1] * batch_size)

    # V = beta2 * V + (1 - beta2) * G * G
    G_square = torch._foreach_mul(G, G)
    G_square = [g.to(dtype=V_dtype) for g in G_square]
    torch._foreach_lerp_(V, G_square, [1 - beta2] * batch_size)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # Compute the denominator for the weight update
    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = torch._foreach_sqrt(V)
    torch._foreach_div_(denom, bias_correction2_sqrt)
    torch._foreach_add_(denom, [epsilon] * batch_size)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1

    # Apply weight decay
    torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Weight update
    # X = X - adj_lr * M / denom
    M_div = torch._foreach_div(M, denom)
    torch._foreach_mul_(M_div, adj_lr)
    torch._foreach_sub_(X, M_div)


@torch.compile(fullgraph=True)
def lion_update_foreach(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
):
    """
    Lion optimizer algorithm (foreach implementation).
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)

    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * G)
    U = torch._foreach_lerp(M, G, [1 - beta1] * batch_size)
    torch._foreach_sign_(U)

    # Update momentum in place with new gradient
    # M = beta2 * M + (1 - beta2) * G
    torch._foreach_lerp_(M, G, [1 - beta2] * batch_size)

    # Apply weight decay
    torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Weight update
    # X = X - lr * U
    torch._foreach_mul_(U, lr)
    torch._foreach_sub_(X, U)


# @torch.compile(fullgraph=True)
def adan_update_foreach(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # First moment buffer (m_t) (modified in place)
    V: List[Tensor],  # Corrected first moment buffer (v_t) (modified in place)
    N: List[Tensor],  # Second moment buffer (modified in place)
    D: List[Tensor],  # Previous grad buffer
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    beta3: Tensor,  # Beta 3 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
):
    """
    Adan optimizer algorithm (foreach implementation).
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)
    assert batch_size == len(V)
    assert batch_size == len(N)

    lr = float(lr)
    beta1 = float(beta1)
    beta2 = float(beta2)
    beta3 = float(beta3)
    weight_decay = float(weight_decay)

    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    torch._foreach_lerp_(M, G, 1 - beta1)

    # v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t - g_{t-1})
    grad_diff = torch._foreach_sub(G, D)  # g_t - g_{t-1}
    torch._foreach_lerp_(V, grad_diff, 1 - beta2)

    # n_t = beta3 * n_{t-1} + (1 - beta3) * [g_t + (1 - beta2)*(g_t - g_{t-1})]^2
    combined_grad = torch._foreach_add(G, torch._foreach_mul(grad_diff, 1 - beta2))
    combined_sq = torch._foreach_mul(combined_grad, combined_grad)
    # torch._foreach_mul_(N, beta3)
    torch._foreach_lerp_(N, combined_sq, 1 - beta3)

    # Bias correction
    bias1 = 1 - math.pow(beta1, step)
    bias2 = 1 - math.pow(beta2, step)
    bias3_sqrt = math.sqrt(1 - math.pow(beta3, step))

    denom = torch._foreach_sqrt(N)
    torch._foreach_div_(denom, bias3_sqrt)
    torch._foreach_add_(denom, epsilon)

    step_size = lr / bias1
    step_size_diff = lr * beta2 / bias2

    # Weight decay (decoupled)
    torch._foreach_div_(X, 1 + lr * weight_decay)

    torch._foreach_addcdiv_(X, M, denom, value=-step_size)
    torch._foreach_addcdiv_(X, V, denom, value=-step_size_diff)

    # Save current grad for next step
    torch._foreach_copy_(D, G)


def adamw_update_foreach_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    step: int,
    epsilon: float,
    cautious_wd: bool = False,
) -> Generator[None, None, None]:
    adamw_update_foreach(
        X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon, # cautious_wd
    )
    yield


def lion_update_foreach_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    cautious_wd: bool = False,
) -> Generator[None, None, None]:
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay, cautious_wd)
    yield