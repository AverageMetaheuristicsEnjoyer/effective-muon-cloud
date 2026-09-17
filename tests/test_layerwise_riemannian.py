import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.llama import Llama
from models.tucker_linear import _mode_product
from optim.layerwise_riemannian import (
    layerwise_tucker_specs, make_layerwise_riemannian,
    retract_layerwise_reference, retract_layerwise_grouped,
)


def tiny_model(variant, kv_heads=4):
    return Llama(SimpleNamespace(
        vocab_size=64, sequence_length=8, n_embd=16, n_head=4, n_kv_head=kv_heads,
        n_layer=3, ffn_hidden_size=32, dropout=0.0, init_std=0.02, rmsnorm_eps=1e-5,
        multiple_of=8, layerwise_execution="reordered", layerwise_tucker_variant=variant,
        layerwise_attention_ranks=(8, 2, 4), layerwise_mlp_ranks=(16, 8, 2 if variant == "A" else 3),
    )).double()


def materialize(spec):
    _, core, factors = spec
    value = core.detach().clone()
    for mode, factor in enumerate(factors):
        value = _mode_product(factor, value, mode)
    return value


class LayerwiseRiemannianTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_initial_gauge_preserves_tensors_and_heads(self):
        for variant in ("A", "B"):
            for kv in (2, 4):
                model = tiny_model(variant, kv)
                specs = layerwise_tucker_specs(model)
                before = [materialize(s) for s in specs]
                retract_layerwise_reference(specs)
                for expected, spec in zip(before, specs):
                    torch.testing.assert_close(materialize(spec), expected, rtol=1e-11, atol=1e-12)
                    for factor in spec[2]:
                        torch.testing.assert_close(factor.T @ factor, torch.eye(factor.shape[1], dtype=factor.dtype), rtol=1e-10, atol=1e-12)

    def compare_grouped(self, method):
        for variant in ("A", "B"):
            reference = tiny_model(variant)
            retract_layerwise_reference(layerwise_tucker_specs(reference))
            grouped = copy.deepcopy(reference)
            ro, rs = make_layerwise_riemannian(reference)
            go, gs = make_layerwise_riemannian(grouped, "grouped", batch_size=2)
            for _ in range(5):
                for a, b in zip(reference.parameters(), grouped.parameters()):
                    a.grad = torch.randn_like(a)
                    b.grad = a.grad.clone()
                ro.step()
                go.step()
                retract_layerwise_reference(rs, ro)
                retract_layerwise_grouped(gs, go, batch_size=2, method=method)
                for a, b in zip(reference.parameters(), grouped.parameters()):
                    torch.testing.assert_close(b, a, rtol=1e-8, atol=1e-10)
                    for key, value in ro.state[a].items():
                        if torch.is_tensor(value):
                            torch.testing.assert_close(go.state[b][key], value, rtol=1e-8, atol=1e-10)
                for _, _, factors in gs:
                    for factor in factors:
                        momentum = go.state[factor]["momentum_buffer"]
                        torch.testing.assert_close(factor.T @ momentum + momentum.T @ factor,
                                                   torch.zeros(factor.shape[1], factor.shape[1], dtype=factor.dtype),
                                                   rtol=0, atol=1e-10)

    def test_grouped_updates_and_transport_match_reference(self):
        self.compare_grouped("qr")

    def test_cholesky_updates_and_transport_match_reference(self):
        self.compare_grouped("cholesky")

    def test_transport_is_gauge_map_differential(self):
        for variant in ("A", "B"):
            model = tiny_model(variant)
            retract_layerwise_reference(layerwise_tucker_specs(model))
            optimizer, specs = make_layerwise_riemannian(model)
            perturbed = copy.deepcopy(model)
            eps = 1e-6
            with torch.no_grad():
                for p, shifted in zip(model.parameters(), perturbed.parameters()):
                    direction = torch.randn_like(p)
                    optimizer.state[p]["momentum_buffer"] = direction
                    shifted.add_(direction, alpha=eps)
            retract_layerwise_reference(specs, optimizer)
            shifted_specs = layerwise_tucker_specs(perturbed)
            retract_layerwise_reference(shifted_specs)
            for (_, core, factors), (_, shifted_core, shifted_factors) in zip(specs, shifted_specs):
                for p, shifted in zip((core, *factors), (shifted_core, *shifted_factors)):
                    torch.testing.assert_close((shifted - p) / eps, optimizer.state[p]["momentum_buffer"], rtol=2e-3, atol=2e-4)

    def test_dense_grouped_muon_matches_production(self):
        from optim.layerwise_muon import make_dense_muon
        from third_party.lite.muonlite import _newton_schulz_impl
        reference = tiny_model(None)
        grouped = copy.deepcopy(reference)
        ro = make_dense_muon(reference)
        go = make_dense_muon(grouped, "grouped", batch_size=2)
        with patch("third_party.lite.muonlite.zeropower_via_newtonschulz5", _newton_schulz_impl):
            for _ in range(3):
                for a, b in zip(reference.parameters(), grouped.parameters()):
                    a.grad = torch.randn_like(a)
                    b.grad = a.grad.clone()
                ro.step()
                go.step()
                for a, b in zip(reference.parameters(), grouped.parameters()):
                    torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-12)
                    if "momentum" in ro.state[a]:
                        torch.testing.assert_close(go.state[b]["momentum"], ro.state[a]["momentum"], rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
