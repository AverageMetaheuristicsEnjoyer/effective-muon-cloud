import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.tucker_linear import TuckerLinear
from optim import spectral_tracking


class _FakeTable:
    def __init__(self, *, columns, data):
        self.columns = columns
        self.data = data


class _FakePlot:
    @staticmethod
    def line_series(*, xs, ys, keys, title, xname):
        return {
            "xs": xs,
            "ys": ys,
            "keys": keys,
            "title": title,
            "xname": xname,
        }


class _FakeWandb:
    def __init__(self):
        self.plot = _FakePlot()
        self.logged = None

    @staticmethod
    def Table(*, columns, data):
        return _FakeTable(columns=columns, data=data)

    def log(self, logs):
        self.logged = logs


def _pure_tucker(in_features, out_features, seed):
    torch.manual_seed(seed)
    return TuckerLinear(
        in_features,
        out_features,
        rank=2,
        bias=False,
        equal_params=False,
        forward_mode="contract",
        dtype=torch.float64,
    )


class SpectralTrackingTest(unittest.TestCase):
    def setUp(self):
        spectral_tracking.reset_spectrum_history()

    def test_fineweb_run_has_exactly_39_post_update_spectral_snapshots(self):
        steps = [
            step
            for step in range(39_250 + 1)
            if spectral_tracking.is_spectral_snapshot_step(step, 1000)
        ]
        self.assertEqual(steps, list(range(1000, 40_000, 1000)))
        self.assertEqual(len(steps), 39)

    def test_every_transformer_block_is_tracked(self):
        self.assertEqual(
            spectral_tracking._tracked_block_indices(12),
            list(range(12)),
        )

    def test_tucker_effective_weight_is_reconstructed_before_svd(self):
        module = _pure_tucker(6, 4, seed=7)
        r1, r2, r3, r4 = module.ranks
        n1, n2 = module.in_modes
        m1, m2 = module.out_modes
        explicit = torch.einsum(
            "ia,jb,cdab,pc,qd->pqij",
            module.U1,
            module.U2,
            module.core_matrix.reshape(r3, r4, r1, r2),
            module.U3,
            module.U4,
        ).reshape(m1 * m2, n1 * n2)

        reconstructed = spectral_tracking._effective_weight(module)
        self.assertEqual(reconstructed.dtype, torch.float32)
        torch.testing.assert_close(
            reconstructed,
            explicit.float(),
            rtol=1e-5,
            atol=1e-6,
        )

        stable_rank, singular_values = (
            spectral_tracking._normalized_stable_rank_and_sv(reconstructed)
        )
        expected_singular_values = torch.linalg.svdvals(explicit.float())
        expected_stable_rank = (
            expected_singular_values.square().sum()
            / expected_singular_values[0].square()
            / min(explicit.shape)
        ).item()
        torch.testing.assert_close(singular_values, expected_singular_values)
        self.assertAlmostEqual(stable_rank, expected_stable_rank, places=6)

    def test_wandb_payload_has_individual_ranks_and_unaveraged_spectra(self):
        block = nn.Module()
        block.attn = nn.Module()
        block.mlp = nn.Module()
        block.attn.q_proj = _pure_tucker(6, 4, seed=11)
        block.attn.k_proj = _pure_tucker(6, 4, seed=12)
        block.attn.v_proj = _pure_tucker(6, 4, seed=13)
        block.attn.o_proj = _pure_tucker(4, 6, seed=14)
        block.mlp.up_proj = _pure_tucker(6, 8, seed=15)
        block.mlp.gate_proj = _pure_tucker(6, 8, seed=16)
        block.mlp.down_proj = _pure_tucker(8, 6, seed=17)
        raw_model = SimpleNamespace(
            transformer=SimpleNamespace(h=[block]),
        )
        fake_wandb = _FakeWandb()

        with patch.object(spectral_tracking, "wandb", fake_wandb):
            spectral_tracking.log_spectrum_and_stable_rank(
                raw_model,
                curr_iter=1000,
                log_stable_rank=True,
                log_full_spectrum=True,
            )
            spectral_tracking.log_spectrum_and_stable_rank(
                raw_model,
                curr_iter=2000,
                log_stable_rank=True,
                log_full_spectrum=True,
            )

        logs = fake_wandb.logged
        self.assertIsNotNone(logs)
        self.assertEqual(logs["iter"], 2000)
        expected_rank_keys = {
            "stable_rank/layer00_q_proj",
            "stable_rank/layer00_k_proj",
            "stable_rank/layer00_v_proj",
            "stable_rank/layer00_o_proj",
            "stable_rank/layer00_up_proj",
            "stable_rank/layer00_gate_proj",
            "stable_rank/layer00_down_proj",
        }
        self.assertEqual(
            {key for key in logs if key.startswith("stable_rank/")},
            expected_rank_keys,
        )
        self.assertEqual(
            {key for key in logs if key.startswith("spectrum/")},
            {
                "spectrum/layer00_q_proj_log_spectrum",
                "spectrum/layer00_k_proj_log_spectrum",
                "spectrum/layer00_v_proj_log_spectrum",
                "spectrum/layer00_o_proj_log_spectrum",
                "spectrum/layer00_up_proj_log_spectrum",
                "spectrum/layer00_gate_proj_log_spectrum",
                "spectrum/layer00_down_proj_log_spectrum",
            },
        )

        expected_q_spectrum = torch.log10(
            torch.linalg.svdvals(
                block.attn.q_proj.materialize_weight(dtype=torch.float32)
            ).clamp_min(spectral_tracking._EPS)
        )
        q_plot = logs["spectrum/layer00_q_proj_log_spectrum"]
        self.assertEqual(
            q_plot["keys"],
            ["step_1000", "step_2000"],
        )
        self.assertEqual(len(q_plot["xs"]), 2)
        self.assertEqual(len(q_plot["ys"]), 2)
        self.assertEqual(
            q_plot["title"],
            "Layer 0 — q_proj log spectrum",
        )
        self.assertEqual(q_plot["xname"], "singular value index")
        logged_q_spectrum = torch.tensor(q_plot["ys"][-1])
        torch.testing.assert_close(
            logged_q_spectrum,
            expected_q_spectrum,
        )

        for key, module in (
            ("up_proj", block.mlp.up_proj),
            ("gate_proj", block.mlp.gate_proj),
            ("down_proj", block.mlp.down_proj),
        ):
            logged_rank = logs[f"stable_rank/layer00_{key}"]
            effective_weight = module.materialize_weight(dtype=torch.float32)
            expected_rank, _ = spectral_tracking._normalized_stable_rank_and_sv(
                effective_weight
            )
            self.assertAlmostEqual(logged_rank, expected_rank, places=6)

    def test_stable_rank_only_snapshot_has_no_spectrum_payload(self):
        block = nn.Module()
        block.attn = nn.Module()
        block.mlp = nn.Module()
        block.attn.q_proj = _pure_tucker(6, 4, seed=21)
        raw_model = SimpleNamespace(
            transformer=SimpleNamespace(h=[block]),
        )
        fake_wandb = _FakeWandb()

        with patch.object(spectral_tracking, "wandb", fake_wandb):
            spectral_tracking.log_spectrum_and_stable_rank(
                raw_model,
                curr_iter=50,
                log_stable_rank=True,
                log_full_spectrum=False,
            )

        logs = fake_wandb.logged
        self.assertIn("stable_rank/layer00_q_proj", logs)
        self.assertFalse(any(key.startswith("spectrum/") for key in logs))

if __name__ == "__main__":
    unittest.main()
