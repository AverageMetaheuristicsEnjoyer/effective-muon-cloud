import copy
import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from models.layerwise_tucker import LayerwiseTuckerAttention, LayerwiseTuckerMLP
from models.llama import Llama, apply_rotary_emb, precompute_freqs_cis


def dense_bank(bank):
    return torch.einsum("di,ej,pc,ijch->phde", bank.hidden, bank.head, bank.role, bank.core)


class LayerwiseTuckerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def compare_gradients(self, actual, expected, inputs):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
        grad = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, inputs, grad, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, inputs, grad)
        for a, e in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, e, rtol=1e-8, atol=1e-10)

    def test_attention_outputs_and_gradients(self):
        for kv_heads in (4, 2):
            for rp in (1, 2):
                with self.subTest(kv_heads=kv_heads, rp=rp):
                    module = LayerwiseTuckerAttention(16, 4, kv_heads, (7, 3, rp)).double()
                    x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
                    q, k, v, oc = module.project_qkv(x)
                    if kv_heads == 4:
                        wq, wk, wv, wo = dense_bank(module.qkvo)
                    else:
                        wq, wo = dense_bank(module.qo)
                        wk, wv = dense_bank(module.kv)
                    qr, kr, vr = [torch.einsum("btd,hde->bthe", x, w) for w in (wq, wk, wv)]
                    for a, e in zip((q, k, v), (qr, kr, vr)):
                        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)
                    # Compare standard causal attention including independent head cores.
                    def attend(q, k, v):
                        return F.scaled_dot_product_attention(
                            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                            is_causal=True, enable_gqa=kv_heads != 4,
                        ).transpose(1, 2)
                    actual = module.project_output(attend(q, k, v), oc)
                    expected = torch.einsum("bthe,hde->btd", attend(qr, kr, vr), wo)
                    self.compare_gradients(actual, expected, (x, *module.parameters()))

    def test_mlp_outputs_and_gradients(self):
        for variant in ("A", "B"):
            for rp in (1, 2):
                with self.subTest(variant=variant, rp=rp):
                    module = LayerwiseTuckerMLP(16, 24, variant, (9, 7, rp)).double()
                    x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
                    weights = torch.einsum("fi,dj,pc,ijc->pfd", module.ff, module.model, module.role, module.core)
                    gate, up = F.linear(x, weights[0]), F.linear(x, weights[1])
                    if variant == "A":
                        down = module.down_out @ module.down_core @ module.down_ff.T
                    else:
                        down = weights[2].T
                    expected = F.linear(F.silu(gate) * up, down)
                    self.compare_gradients(module(x), expected, (x, *module.parameters()))

    def test_full_model_training_and_layer_independence(self):
        for variant in ("A", "B"):
            for kv_heads in (4, 2):
                config = SimpleNamespace(
                    vocab_size=64, sequence_length=8, n_embd=16, n_head=4,
                    n_kv_head=kv_heads, n_layer=2, dropout=0.0, init_std=0.02,
                    rmsnorm_eps=1e-5, ffn_hidden_size=24, multiple_of=8,
                    layerwise_tucker_variant=variant, layerwise_attention_ranks=(8, 2, 2),
                    layerwise_mlp_ranks=(12, 8, 2),
                )
                model = Llama(config)
                first, second = model.transformer.h
                self.assertTrue({id(p) for p in first.parameters()}.isdisjoint(
                    {id(p) for p in second.parameters()}))
                for block in (first, second):
                    self.assertFalse(any(isinstance(m, torch.nn.Linear) for m in block.modules()))
                tokens = torch.randint(64, (2, 8))
                before = model(tokens, tokens)["loss"]
                before.backward()
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                optimizer.step()
                self.assertTrue(torch.isfinite(model(tokens, tokens)["loss"]))
                model.eval()
                with torch.no_grad():
                    changed = tokens.clone()
                    changed[:, 4:] = torch.randint(64, (2, 4))
                    a = model(tokens, tokens, get_logits=True)["logits"]
                    b = model(changed, changed, get_logits=True)["logits"]
                    torch.testing.assert_close(a[:, :4], b[:, :4])

    def test_reordered_mlp_gradients_and_updates(self):
        for variant in ("A", "B"):
            for rp in (1, 2):
                with self.subTest(variant=variant, rp=rp):
                    ref = LayerwiseTuckerMLP(16, 24, variant, (9, 7, rp)).double()
                    opt = copy.deepcopy(ref)
                    opt.execution = "reordered"
                    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3) for m in (ref, opt)]
                    for _ in range(3):
                        x = torch.randn(2, 5, 16, dtype=torch.float64)
                        grad = torch.randn_like(x)
                        xr, xo = x.clone().requires_grad_(), x.clone().requires_grad_()
                        yr, yo = ref(xr), opt(xo)
                        torch.testing.assert_close(yr, yo, rtol=1e-9, atol=1e-11)
                        yr.backward(grad)
                        yo.backward(grad)
                        torch.testing.assert_close(xr.grad, xo.grad, rtol=1e-8, atol=1e-11)
                        for pr, po in zip(ref.parameters(), opt.parameters()):
                            torch.testing.assert_close(pr.grad, po.grad, rtol=1e-8, atol=1e-11)
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                        for pr, po in zip(ref.parameters(), opt.parameters()):
                            torch.testing.assert_close(pr, po, rtol=1e-8, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
