import torch
from torch import Tensor

@torch.compile(fullgraph=True)
def zeropower_via_newtonschulz5_jordan(G: Tensor, epsilon: float = 1e-7):
    """
    Newton-Schulz iteration to approximate the orthogonalization of X.
    """
    # Newton-Schulz constants
    ns_consts = [
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


from itertools import repeat
import torch
coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933) ,
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601) ,
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137) ,
    (3.3184196573706015, -2.488488024314874, 0.51004894012372) ,
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673) ,
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835) ,
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248) ,
    (1.875, -1.25, 0.375), # subsequent coeffs equal this numerically
]
# safety factor for numerical stability (but exclude last polynomial )
coeffs_list = [( a / 1.01, b / 1.01**3, c / 1.01**5)
    for (a, b, c ) in coeffs_list [: -1]] +[ coeffs_list[ -1]]


coeffs_list_2 = [
    (7.2086, -15.5131, 9.0178),
    (3.9623, -2.5813, 0.4542),
    (3.9466, -2.5765, 0.4544),
    (3.8991, -2.5671, 0.4566),
    (3.7186, -2.5308, 0.4653),
    (3.1390, -2.3073, 0.4733),
    (2.1715, -1.5246, 0.3885),
    (1.8648, -1.2224, 0.3577),
]

@torch.compile
def PolarExpressOrig(G: torch.Tensor, steps: int, epsilon=None) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16() # for speed
    if G.size(-2) > G.size(-1): X = X.mT # this reduces FLOPs
    
    X = X / (X.norm(dim=(-2, -1), keepdim = True) * 1.01)
    hs = coeffs_list[:steps] + list(
        repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs :
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X # X <- aX + bX ˆ3 + cX ˆ5
    if G.size(-2) > G.size(-1): X = X.mT
    return X


# https://github.com/pytorch/pytorch/issues/148819#issuecomment-3077425085
@torch.compile
def PolarExpressModified(G: torch.Tensor, steps: int, epsilon=1e-7) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16() # for speed
    if G.size(-2) > G.size(-1): X = X.mT # this reduces FLOPs
    
    X = X / (X.norm(dim=(-2, -1), keepdim = True) + epsilon)
    hs = coeffs_list_2[:steps] + list(
        repeat(coeffs_list_2[-1], steps - len(coeffs_list_2)))
    for a, b, c in hs :
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X # X <- aX + bX ˆ3 + cX ˆ5
    if G.size(-2) > G.size(-1): X = X.mT
    return X

@torch.compile
def zeropower_5777_left_1e_3(G: torch.Tensor, epsilon=1e-7) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16() # for speed
    if G.size(-2) > G.size(-1): X = X.mT # this reduces FLOPs
    
    X = X / (X.norm(dim=(-2, -1), keepdim = True) + epsilon)
    
    coefs = [
        (7.490273393478118, -20.90656923658714, 15.026559337174017),
        (5.011059931358244, -6.900580758447274, 3.101882145128696, -0.41882111943633665),
        (5.087767907791593, -7.222353263169536, 3.3466767497580996, -0.46581389194702894),
        (3.6895604572396943, -4.9833872244791495, 2.527541407733747, -0.39914443187258075)
    ]
    a, b, c = coefs[0]
    A = X @ X.mT
    B = b * A + c * A @ A
    X = a * X + B @ X # X <- aX + bXˆ3 + cXˆ5

    for a, b, c, d in coefs[1:]:
        A = X @ X.mT
        B = c * A + d * A @ A
        C = b * A + B @ A
        X = a * X + C @ X # X <- aX + bXˆ3 + cXˆ5 + dXˆ7
    if G.size(-2) > G.size(-1): X = X.mT
    return X

@torch.compile
def zeropower_5779_left_15e_4(G: torch.Tensor, epsilon=1e-7) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16() # for speed
    if G.size(-2) > G.size(-1): X = X.mT # this reduces FLOPs
    
    X = X / (X.norm(dim=(-2, -1), keepdim = True) + epsilon)
    
    coefs = [
        (7.490273393478118, -20.90656923658714, 15.026559337174017),
        (5.020496369966255, -6.939638128135302, 3.131198466743433, -0.4243732464303429),
        (4.983538423583185, -7.122583345071286, 3.351818783586265, -0.4747333038380677),
        (3.612721861510107, -6.88155563900719, 6.2102586208887125, -2.3822634574717028, 0.3221095115956162)
    ]
    a, b, c = coefs[0]
    A = X @ X.mT
    B = b * A + c * A @ A
    X = a * X + B @ X # X <- aX + bXˆ3 + cXˆ5

    for a, b, c, d in coefs[1:-1]:
        A = X @ X.mT
        B = c * A + d * A @ A
        C = b * A + B @ A
        X = a * X + C @ X # X <- aX + bXˆ3 + cXˆ5 + dXˆ7
    
    a, b, c, d, e = coefs[-1]
    A = X @ X.mT
    B = d * A + e * A @ A
    C = c * A + B @ A
    D = b * A + C @ A
    X = a * X + D @ X # X <- aX + bXˆ3 + cXˆ5 + dXˆ7 + eXˆ9
    if G.size(-2) > G.size(-1): X = X.mT
    return X


    