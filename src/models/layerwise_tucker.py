import math

import torch
from torch import nn
from torch.nn import functional as F


class AttentionBank(nn.Module):
    def __init__(self, d, head_dim, roles, heads, rd, rh, rp, init_std):
        super().__init__()
        self.hidden = nn.Parameter(torch.empty(d, rd))
        self.head = nn.Parameter(torch.empty(head_dim, rh))
        self.role = nn.Parameter(torch.empty(roles, rp))
        self.core = nn.Parameter(torch.empty(rd, rh, rp, heads))
        nn.init.normal_(self.hidden, std=1 / math.sqrt(rd))
        nn.init.normal_(self.head, std=1 / math.sqrt(rh))
        nn.init.orthogonal_(self.role)
        nn.init.normal_(self.core, std=init_std * math.sqrt(roles / rp))

    def role_cores(self):
        return torch.einsum("pc,ijch->phij", self.role, self.core)

    def project(self, latent, cores):
        projected = F.linear(latent, cores.permute(0, 1, 3, 2).flatten(0, 2))
        projected = projected.unflatten(-1, (*cores.shape[:2], cores.shape[-1]))
        return F.linear(projected, self.head)

    def output(self, z, core):
        latent = z @ self.head
        latent = F.linear(latent.flatten(-2), core.permute(1, 0, 2).flatten(1))
        return F.linear(latent, self.hidden)


class LayerwiseTuckerAttention(nn.Module):
    def __init__(self, d, heads, kv_heads, ranks, init_std=0.02):
        super().__init__()
        rd, rh, rp = ranks
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = d // heads
        if heads == kv_heads:
            self.qkvo = AttentionBank(d, self.head_dim, 4, heads, rd, rh, rp, init_std)
        else:
            self.qo = AttentionBank(d, self.head_dim, 2, heads, rd, rh, min(rp, 2), init_std)
            self.kv = AttentionBank(d, self.head_dim, 2, kv_heads, rd, rh, min(rp, 2), init_std)

    def project_qkv(self, x):
        if self.heads == self.kv_heads:
            cores = self.qkvo.role_cores()
            qkv = self.qkvo.project(x @ self.qkvo.hidden, cores[:3])
            q, k, v = qkv.unbind(dim=-3)
            output_core = cores[3]
        else:
            qo = self.qo.role_cores()
            q = self.qo.project(x @ self.qo.hidden, qo[:1]).squeeze(-3)
            k, v = self.kv.project(x @ self.kv.hidden, self.kv.role_cores()).unbind(dim=-3)
            output_core = qo[1]
        return q, k, v, output_core

    def project_output(self, z, output_core):
        bank = self.qkvo if self.heads == self.kv_heads else self.qo
        return bank.output(z, output_core)


class LayerwiseTuckerMLP(nn.Module):
    def __init__(self, d, ff, variant, ranks, init_std=0.02):
        super().__init__()
        rff, rm, rp = ranks
        roles = 2 if variant == "A" else 3
        self.variant = variant
        self.ff = nn.Parameter(torch.empty(ff, rff))
        self.model = nn.Parameter(torch.empty(d, rm))
        self.role = nn.Parameter(torch.empty(roles, rp))
        self.core = nn.Parameter(torch.empty(rff, rm, rp))
        nn.init.normal_(self.ff, std=1 / math.sqrt(rff))
        nn.init.normal_(self.model, std=1 / math.sqrt(rm))
        nn.init.orthogonal_(self.role)
        nn.init.normal_(self.core, std=init_std * math.sqrt(roles / rp))
        if variant == "A":
            self.down_out = nn.Parameter(torch.empty(d, rm))
            self.down_ff = nn.Parameter(torch.empty(ff, rff))
            self.down_core = nn.Parameter(torch.empty(rm, rff))
            nn.init.normal_(self.down_out, std=1 / math.sqrt(rm))
            nn.init.normal_(self.down_ff, std=1 / math.sqrt(rff))
            nn.init.normal_(self.down_core, std=init_std)

    def forward(self, x):
        cores = torch.einsum("pc,ijc->pij", self.role, self.core)
        gu = F.linear(x @ self.model, cores[:2].flatten(0, 1))
        gu = F.linear(gu.unflatten(-1, (2, self.ff.shape[1])), self.ff)
        gate, up = gu.unbind(dim=-2)
        h = F.silu(gate) * up
        if self.variant == "A":
            return F.linear(F.linear(h @ self.down_ff, self.down_core), self.down_out)
        return F.linear((h @ self.ff) @ cores[2], self.model)
