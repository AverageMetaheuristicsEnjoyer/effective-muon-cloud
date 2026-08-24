import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import patch

from scripts.tucker_benchmark.common import (
    accumulation_steps,
    requested_controls,
    result_matches_request,
    summarize,
)


class TuckerBenchmarkTest(unittest.TestCase):
    def args(self):
        return SimpleNamespace(
            sequence_length=1024,
            warmup_steps=3,
            measured_steps=12,
            lr=1e-3,
            momentum=0.95,
            beta1=0.9,
            beta2=0.99,
            weight_decay=0.1,
            eps=1e-8,
            grad_clip=1.0,
            seed=0,
            exclusive_gpu=True,
        )

    def test_accumulation_preserves_tokens_per_step(self):
        self.assertEqual(accumulation_steps(16_384, 1, 1024), 16)
        self.assertEqual(accumulation_steps(16_384, 16, 1024), 1)
        with self.assertRaises(ValueError):
            accumulation_steps(16_384, 3, 1024)

    def test_summary(self):
        result = summarize([1.0, 2.0, 3.0])
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["count"], 3)

    def test_result_reuse_requires_identical_controls(self):
        controls = requested_controls(self.args(), 4, 4)
        self.assertEqual(controls["tucker_forward_mode"], "contract")
        payload = {
            "status": "complete",
            "variant": {"name": "static_tucker"},
            "benchmark": controls,
            "samples": [{}] * 12,
        }
        self.assertTrue(
            result_matches_request(payload, "static_tucker", controls)
        )
        changed = dict(controls, seed=1)
        self.assertFalse(
            result_matches_request(payload, "static_tucker", changed)
        )

    @unittest.skipUnless(find_spec("torch"), "PyTorch is not installed")
    def test_static_tucker_step_never_materializes_weight(self):
        import torch

        from models.tucker_linear import TuckerLinear, retract_tucker_modules_
        from models.utils import get_model
        from scripts.tucker_benchmark.benchmark_train_step import (
            build_tucker_optimizer,
            make_config,
        )

        dense_args = self.args()
        dense_args.variant = "dense_adamw"
        dense_args.microbatch = 2
        dense_config = make_config(dense_args)
        self.assertFalse(dense_config.tucker_retract_every_step)
        self.assertFalse(dense_config.tucker_vector_transport)

        args = self.args()
        args.variant = "static_tucker"
        args.microbatch = 2
        config = make_config(args)
        config.vocab_size = 32
        config.sequence_length = 4
        config.n_layer = 1
        config.n_embd = 8
        config.n_head = 2
        config.multiple_of = 4
        config.ffn_hidden_size = 16
        config.tucker_rank = "2"
        config.target_parameter_count = 0
        config.target_parameter_tolerance = 0

        model = get_model(config)
        modules = [
            module for module in model.modules() if isinstance(module, TuckerLinear)
        ]
        self.assertTrue(modules)
        self.assertEqual({module.resolved_forward_mode for module in modules}, {"contract"})
        optimizer, _ = build_tucker_optimizer(args, model)
        inputs = torch.randint(0, config.vocab_size, (2, config.sequence_length))
        targets = torch.randint(0, config.vocab_size, (2, config.sequence_length))

        with patch.object(
            TuckerLinear,
            "materialize_weight",
            side_effect=AssertionError("materialize_weight was called"),
        ):
            loss = model(inputs, targets=targets)["loss"]
            loss.backward()
            optimizer.step()
            retract_tucker_modules_(
                model,
                optimizer=optimizer,
                transport_optimizer_state=True,
            )

        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
