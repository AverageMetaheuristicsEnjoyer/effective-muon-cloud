# ./sota_opt/swan/swan.py

import torch
from torch.optim import Optimizer
from typing import Iterable, Literal
from torch import nn

# Import efficient Newton-Schulz implementations
from ..dion.newton_schulz_funcs import (
    zeropower_via_newtonschulz5_jordan,
    PolarExpressOrig,
    PolarExpressModified,
    zeropower_5777_left_1e_3,
    zeropower_5779_left_15e_4,
)

try:
    from ..dion.newton_schulz_triton import newton_schulz_triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    newton_schulz_triton = None


class SWAN(Optimizer):
    """
    SWAN (Stochastic Whitening Accumulated gradieNt) Optimizer
    
    Based on: https://arxiv.org/pdf/2412.13148v2
    
    Applies gradient normalization and whitening before the update step.
    Compatible with momentum, nesterov, and other SGD features.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0)
        dampening: dampening for momentum (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        nesterov: enables Nesterov momentum (default: False)
        sign_update: use sign of gradient for update (default: False)
        
        # SWAN-specific parameters
        k: number of Newton-Schulz iterations for whitening (default: 10)
            Only used if ns_func='manual'
        beta: step size for Newton-Schulz iterations (default: 0.8)
            Only used if ns_func='manual'
        rescale: rescale whitened gradient to match original norm (default: True)
        min_numel_whitening: minimum parameter size to apply whitening (default: 1)
        ns_func: Newton-Schulz function to use (default: 'jordan')
            Options: 'manual', 'jordan', 'polar_orig', 'polar_mod', '5777', '5779', 'triton'
        epsilon: small constant for numerical stability (default: 1e-7)
    """
    
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        sign_update: bool = False,
        # SWAN-specific parameters
        k: int = 5,
        beta: float = 0.8,
        rescale: bool = True,
        min_numel_whitening: int = 1,
        ns_func: Literal['manual', 'jordan', 'polar_orig', 'polar_mod', '5777', '5779', 'triton'] = 'jordan',
        epsilon: float = 1e-7,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if k < 1:
            raise ValueError(f"Invalid k value: {k}")
        if beta <= 0.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if ns_func == 'triton' and not TRITON_AVAILABLE:
            raise ValueError("Triton is not available. Please install triton or use a different ns_func.")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            sign_update=sign_update,
            k=k,
            beta=beta,
            rescale=rescale,
            min_numel_whitening=min_numel_whitening,
            ns_func=ns_func,
            epsilon=epsilon,
        )
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('sign_update', False)
            group.setdefault('k', 10)
            group.setdefault('beta', 0.8)
            group.setdefault('rescale', True)
            group.setdefault('min_numel_whitening', 1)
            group.setdefault('ns_func', 'jordan')
            group.setdefault('epsilon', 1e-7)

    @torch.no_grad()
    def grad_norm(self, G):
        """
        GradNorm operator (Algorithm 3)
        
        Normalizes gradient by:
        1. Subtracting row-wise mean (center each row)
        2. Dividing by row-wise standard deviation
        
        Args:
            G: gradient tensor of shape (m, n)
        
        Returns:
            Normalized gradient
        """
        if G.dim() == 1:
            # For 1D tensors, simple standardization
            mean = G.mean()
            std = G.std(unbiased=False).clamp(min=1e-8)
            return (G - mean) / std
        
        # G is m x n
        # Compute row-wise mean: (1/n) ∑_j G_{:,j}
        row_mean = G.mean(dim=1, keepdim=True)  # (m, 1)
        
        # Center: G - mean
        G_centered = G - row_mean
        
        # Compute row-wise standard deviation
        # sqrt((1/n) sum_j (G_{:,j} - mean)^2)
        row_std = G_centered.pow(2).mean(dim=1, keepdim=True).sqrt()  # (m, 1)
        
        # Avoid division by zero
        row_std = row_std.clamp(min=1e-8)
        
        return G_centered / row_std

    @torch.no_grad()
    def grad_whitening_manual(self, G, k, beta):
        """
        GradWhitening operator (Algorithm 2) - Manual implementation
        
        Uses Newton-Schulz iteration to compute (GG^T)^{-1/2} @ G
        This decorrelates the gradient components.
        
        Args:
            G: gradient tensor of shape (m, n) where m <= n
            k: number of Newton-Schulz iterations
            beta: step size for iterations
        
        Returns:
            Whitened gradient
        """
        if G.dim() == 1:
            # For 1D tensors, return as is
            return G
        
        m, n = G.shape
        
        # Handle case where m > n by working with transpose
        if m > n:
            G_work = G.t()  # Now (n, m) where n < m
            m, n = n, m
            transposed = True
        else:
            G_work = G
            transposed = False
        
        # Compute Y = GG^T (m x m matrix)
        Y = torch.mm(G_work, G_work.t())
        
        # Initialize Z = I (identity matrix)
        Z = torch.eye(m, device=G.device, dtype=G.dtype)
        I = Z.clone()  # Keep a copy of identity for reuse
        
        # Newton-Schulz iterations to approximate Z ≈ (GG^T)^{-1/2}
        for _ in range(k):
            # Compute ZY once for efficiency
            ZY = torch.mm(Z, Y)
            
            # Compute 3I - ZY once for efficiency
            three_I_minus_ZY = 3.0 * I - ZY
            
            # Y ← beta * Y * (3I - ZY)
            Y = beta * torch.mm(Y, three_I_minus_ZY)
            
            # Z ← beta * (3I - ZY) * Z
            Z = beta * torch.mm(three_I_minus_ZY, Z)
        
        # Apply whitening: ZG
        result = torch.mm(Z, G_work)
        
        # Transpose back if needed
        if transposed:
            result = result.t()
        
        return result

    @torch.no_grad()
    def grad_whitening(self, G, ns_func, k=None, epsilon=1e-7):
        """
        GradWhitening using efficient Newton-Schulz implementations
        
        Args:
            G: gradient tensor
            ns_func: which Newton-Schulz function to use
            k: number of iterations (only for 'polar_orig' and 'polar_mod')
            epsilon: numerical stability constant
        
        Returns:
            Whitened gradient
        """
        if G.dim() == 1:
            return G
        
        # Select the appropriate Newton-Schulz function
        if ns_func == 'jordan':
            return zeropower_via_newtonschulz5_jordan(G, epsilon=epsilon)
        elif ns_func == 'polar_orig':
            steps = k if k is not None else 8
            return PolarExpressOrig(G, steps=steps, epsilon=epsilon)
        elif ns_func == 'polar_mod':
            steps = k if k is not None else 8
            return PolarExpressModified(G, steps=steps, epsilon=epsilon)
        elif ns_func == '5777':
            return zeropower_5777_left_1e_3(G, epsilon=epsilon)
        elif ns_func == '5779':
            return zeropower_5779_left_15e_4(G, epsilon=epsilon)
        elif ns_func == 'triton':
            return newton_schulz_triton(G, epsilon=epsilon)
        else:
            raise ValueError(f"Unknown ns_func: {ns_func}")

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        
        Returns:
            Loss if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Extract SWAN-specific hyperparameters
            k = group['k']
            beta = group['beta']
            rescale = group['rescale']
            min_numel = group['min_numel_whitening']
            ns_func = group['ns_func']
            epsilon = group['epsilon']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Apply weight decay (L2 regularization)
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Store original shape for later
                orig_shape = grad.shape
                orig_dtype = grad.dtype
                
                # Reshape to 2D for SWAN processing
                if grad.dim() > 2:
                    # Flatten all dimensions except the first
                    grad_2d = grad.view(grad.shape[0], -1)
                elif grad.dim() == 1:
                    # For 1D, we'll skip whitening but still apply GradNorm if large enough
                    grad_2d = grad.unsqueeze(0) if grad.numel() >= min_numel else grad
                else:
                    grad_2d = grad

                # Apply SWAN transformation only if parameter is large enough and multi-dimensional
                if grad.numel() >= min_numel and grad.dim() >= 2:
                    # Step 1: Apply GradNorm (Algorithm 3)
                    grad_normed = self.grad_norm(grad_2d)
                    
                    # Store original norm for optional rescaling
                    if rescale:
                        orig_norm = grad_normed.norm()
                    
                    # Step 2: Apply GradWhitening (Algorithm 2)
                    if ns_func == 'manual':
                        grad_whitened = self.grad_whitening_manual(grad_normed, k, beta)
                    else:
                        grad_whitened = self.grad_whitening(grad_normed, ns_func, k, epsilon)
                    
                    # Step 3: Rescale to match original norm (optional)
                    if rescale:
                        whitened_norm = grad_whitened.norm()
                        if whitened_norm > epsilon:
                            grad_whitened = grad_whitened * (orig_norm / whitened_norm)
                    
                    # Reshape back to original shape and dtype
                    grad_processed = grad_whitened.view(orig_shape).to(orig_dtype)
                else:
                    # For small or 1D parameters, skip SWAN transformation
                    grad_processed = grad

                # Apply momentum if specified
                if group['momentum'] != 0:
                    if 'momentum_buffer' not in state:
                        # Initialize momentum buffer
                        buf = state['momentum_buffer'] = torch.clone(grad_processed).detach()
                    else:
                        buf = state['momentum_buffer']
                        # Update momentum buffer
                        buf.mul_(group['momentum']).add_(grad_processed, alpha=1 - group['dampening'])

                    if group['nesterov']:
                        # Nesterov momentum
                        grad_processed = grad_processed.add(buf, alpha=group['momentum'])
                    else:
                        # Standard momentum
                        grad_processed = buf

                # Apply sign update if specified
                if group['sign_update']:
                    grad_processed = grad_processed.sign()

                # Apply the update: W^(t) = W^(t-1) - eta * delta_W
                p.add_(grad_processed, alpha=-group['lr'])

        return loss
