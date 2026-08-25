"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from .tensorized_attention import tensorized_attention_from_config
from .tucker_linear import TuckerLinear


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(config.sequence_length, config.sequence_length)
                ).view(1, 1, config.sequence_length, config.sequence_length),
            )

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        (
            B,
            T,
            C,
        ) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, nh, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config, exp_factor=1.0):
        super().__init__()
        self.dim_exp_factor = exp_factor * 4

        hidden_size = getattr(config, "ffn_hidden_size", 0)
        if hidden_size <= 0:
            hidden_size = int(self.dim_exp_factor * config.n_embd)

        self.c_fc = nn.Linear(
            config.n_embd,
            hidden_size,
            bias=config.bias,
        )
        self.c_proj = nn.Linear(
            hidden_size,
            config.n_embd,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x, {}


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = (
            tensorized_attention_from_config(config)
            if getattr(config, "attention_type", "standard") == "tensorized"
            else CausalSelfAttention(config)
        )
        self.parallel = config.parallel_block
        if not self.parallel:
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, *args, **kwargs):
        if self.parallel:
            # from GPT-J 6B https://github.com/kingoflolz/mesh-transformer-jax/blob/f8315e3003033b23f21d78361b288953064e0e76/mesh_transformer/layers.py#L299
            x_ln = self.ln_1(x, *args, **kwargs)
            x_attn = self.attn(x_ln)
            x_ffn, _ = self.mlp(x_ln)
            x = x + x_attn + x_ffn
        else:
            x = x + self.attn(self.ln_1(x, *args, **kwargs))
            x_, _ = self.mlp(self.ln_2(x, *args, **kwargs))
            x = x + x_
        return x


class GPTBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self._tokenizer = None

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.sequence_length, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("o_proj.weight"):
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=self.config.init_std / math.sqrt(2 * config.n_layer),
                )

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = tiktoken.get_encoding("gpt2")
        return self._tokenizer

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    @property
    def num_fwd_flops(self):
        if not hasattr(self, '_num_fwd_flops'):
            # Embeddings are lookups in the forward pass, not matmuls — exclude them.
            n_params_no_emb = sum(
                p.numel() for name, p in self.named_parameters()
                if 'wte' not in name and 'wpe' not in name
            )
            # 2 FLOPs per MAC. Linear projections are already covered by the
            # parameter term; account separately for the attention contractions.
            if getattr(self.config, "attention_type", "standard") == "tensorized":
                rank = self.config.tensorized_rank
                if self.config.tensorized_mode == "split_concat":
                    attention_flops = 2 * rank * self.config.sequence_length ** 2
                else:
                    attention_flops = 2 * rank * self.config.sequence_length
            else:
                # 2*2 for Q*K^T and out*V per standard attention layer.
                attention_flops = 4 * self.config.n_embd * self.config.sequence_length
            estimate = (
                2 * n_params_no_emb
                + self.config.n_layer * attention_flops
            )
            # ``2 * parameter_count`` assumes one dense MAC per weight. Tucker
            # factors are reused across mode contractions, so replace that term
            # with the module's actual contraction count.
            for module in self.modules():
                if isinstance(module, TuckerLinear):
                    estimate += (
                        module.forward_flops_per_token
                        - 2 * module.weight_parameter_count
                    )
            self._num_fwd_flops = estimate
        return self._num_fwd_flops

    @property
    def num_bck_flops(self):
        if not hasattr(self, '_num_bck_flops'):
            n_params_no_emb = sum(
                p.numel() for name, p in self.named_parameters()
                if 'wte' not in name and 'wpe' not in name
            )
            if getattr(self.config, "attention_type", "standard") == "tensorized":
                rank = self.config.tensorized_rank
                if self.config.tensorized_mode == "split_concat":
                    attention_flops = 4 * rank * self.config.sequence_length ** 2
                else:
                    attention_flops = 4 * rank * self.config.sequence_length
            else:
                attention_flops = 8 * self.config.n_embd * self.config.sequence_length
            # Backward is ~2x forward for weights + ~2x for input grads = 4x params
            estimate = (
                4 * n_params_no_emb
                + self.config.n_layer * attention_flops
            )
            for module in self.modules():
                if isinstance(module, TuckerLinear):
                    estimate += (
                        2 * module.forward_flops_per_token
                        - 4 * module.weight_parameter_count
                    )
            self._num_bck_flops = estimate
        return self._num_bck_flops

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(self, idx, targets=None, get_logits=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                label_smoothing=getattr(self.config, "label_smoothing", 0.0),
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None
        logits = logits if get_logits else None
        return {
            "logits": logits,
            "loss": loss,
        }

    def crop_sequence_length(self, sequence_length):
        # model surgery to decrease the block size if necessary
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:sequence_length]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[
                    :, :, :sequence_length, :sequence_length
                ]

    def from_pretrained(self, model_path):
        paths = model_path.split(",")
        if len(paths) == 1:
            loaded_state = torch.load(
                str(model_path + "/ckpt.pt"),
                map_location=torch.device(self.config.device),
            )
            state_to_load = loaded_state["model"]
            state_to_load = {
                ".".join(k.split(".")[1:]): v  # drop _orig_mod from keys
                for k, v in state_to_load.items()
            }

    def get_parameter_group_specs(self):
        """
        Separate parameters into weight-decayed and non-decayed groups.
        Linear weights are decayed; biases, norms, and embeddings are not.
        """
        decay = set()
        decay_non_projection = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        from .utils import BLACKLIST_WEIGHT_MODULES

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn == "core_logits":
                    # Diagonal BTD core coefficients behave like attention
                    # logits, not a dense projection matrix.
                    no_decay.add(fpn)
                elif isinstance(m, TuckerLinear):
                    # Tucker factors/core/residual replace a decayed Linear
                    # weight. The Llama lm_head follows the repo's existing
                    # no-decay treatment of the output projection.
                    if fpn.startswith("lm_head."):
                        no_decay.add(fpn)
                    elif pn == "residual_tail":
                        # This is a decayed projection-weight remainder, but it
                        # is 1-D and must not be handed to matrix optimizers.
                        decay_non_projection.add(fpn)
                    else:
                        decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, BLACKLIST_WEIGHT_MODULES):
                    no_decay.add(fpn)

        # GPTBase ties lm_head.weight to wte.weight; Llama historically follows
        # the same no-decay grouping for its independent output head.
        decay.discard("lm_head.weight")
        if "lm_head.weight" in dict(self.named_parameters()):
            no_decay.add("lm_head.weight")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = (
            (decay & no_decay)
            | (decay & decay_non_projection)
            | (decay_non_projection & no_decay)
        )
        union_params = decay | decay_non_projection | no_decay
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        groups = [
            {"params": sorted(list(decay)), "is_proj_params": True},
        ]
        if decay_non_projection:
            groups.append(
                {
                    "params": sorted(list(decay_non_projection)),
                    "is_proj_params": False,
                }
            )
        groups.append(
            {
                "params": sorted(list(no_decay)),
                "weight_decay": 0.0,
                "is_proj_params": False,
            }
        )
        return groups

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.sequence_length
                else idx[:, -self.config.sequence_length :]
            )
            logits = self(idx_cond, get_logits=True)["logits"]
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = (
            torch.tensor(
                self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})
            )
            .view(1, -1)
            .to(next(self.lm_head.parameters()).device)
        )
        out_idx = (
            self.generate(idx, max_new_tokens, temperature, top_k)
            .view(-1)
            .to("cpu")
            .numpy()
        )
        return self.tokenizer.decode(out_idx)
