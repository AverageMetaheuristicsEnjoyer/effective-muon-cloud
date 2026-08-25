import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.tensorized_attention import TensorizedAttention


class TensorizedAttentionTest(unittest.TestCase):
    def _make(self, mode, *, causal=True, chunk_size=2):
        torch.manual_seed(7)
        return TensorizedAttention(
            d_model=8,
            rank=3,
            num_cores=2,
            max_sequence_length=6,
            mode=mode,
            causal=causal,
            dropout=0.0,
            query_chunk_size=chunk_size,
        )

    def test_forward_backward_in_both_modes(self):
        for mode in TensorizedAttention.MODES:
            with self.subTest(mode=mode):
                module = self._make(mode)
                x = torch.randn(2, 5, 8, requires_grad=True)
                output = module(x)

                self.assertEqual(output.shape, (2, 5, 8))
                output.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertIsNotNone(module.core_logits.grad)
                self.assertGreater(module.core_logits.grad.abs().sum().item(), 0.0)

    def test_causal_output_does_not_depend_on_future_tokens(self):
        for mode in TensorizedAttention.MODES:
            with self.subTest(mode=mode):
                module = self._make(mode, causal=True).eval()
                original = torch.randn(2, 6, 8)
                changed = original.clone()
                changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 20.0

                original_output = module(original)
                changed_output = module(changed)
                torch.testing.assert_close(
                    original_output[:, :3],
                    changed_output[:, :3],
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_split_concat_chunking_is_numerically_equivalent(self):
        chunked = self._make("split_concat", chunk_size=2).eval()
        unchunked = self._make("split_concat", chunk_size=6).eval()
        unchunked.load_state_dict(chunked.state_dict())
        x = torch.randn(2, 6, 8)

        torch.testing.assert_close(
            chunked(x),
            unchunked(x),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_shorter_sequences_use_the_configured_projection(self):
        for mode in TensorizedAttention.MODES:
            with self.subTest(mode=mode):
                module = self._make(mode)
                self.assertEqual(module(torch.randn(1, 4, 8)).shape, (1, 4, 8))

    def test_rejects_sequences_above_configured_maximum(self):
        module = self._make("reconstruction")
        with self.assertRaisesRegex(ValueError, "exceeds configured maximum"):
            module(torch.randn(1, 7, 8))

    def test_gpt_training_integration_and_parameter_groups(self):
        from models.base import GPTBase

        config = SimpleNamespace(
            vocab_size=32,
            sequence_length=5,
            n_embd=8,
            n_head=1,
            n_layer=1,
            dropout=0.0,
            bias=False,
            parallel_block=False,
            init_std=0.02,
            device="cpu",
            attention_type="tensorized",
            tensorized_mode="split_concat",
            tensorized_rank=3,
            tensorized_num_cores=2,
            tensorized_causal=True,
            tensorized_query_chunk_size=2,
            ffn_hidden_size=16,
            label_smoothing=0.1,
        )
        model = GPTBase(config)
        tokens = torch.randint(0, config.vocab_size, (2, config.sequence_length))
        result = model(tokens, targets=tokens, get_logits=True)

        self.assertEqual(result["logits"].shape, (2, 5, 32))
        result["loss"].backward()
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertIsNotNone(
            model.transformer.h[0].attn.core_logits.grad
        )

        groups = model.get_parameter_group_specs()
        grouped_names = set().union(*(set(group["params"]) for group in groups))
        self.assertEqual(grouped_names, set(dict(model.named_parameters())))


if __name__ == "__main__":
    unittest.main()
