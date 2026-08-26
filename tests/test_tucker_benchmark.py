import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import patch

from scripts.tucker_benchmark.common import (
    MODEL_SPECS,
    accumulation_steps,
    requested_controls,
    result_matches_request,
    summarize,
    tucker_rank_plan,
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
            model_size="257m",
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
        self.assertEqual(controls["tucker_rank_multiple"], 8)
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

    def test_rank8_plans_are_iso_param_at_every_scale(self):
        for spec in MODEL_SPECS:
            with self.subTest(model=spec["name"]):
                plan, parameters = tucker_rank_plan(spec["name"])
                self.assertTrue(plan)
                self.assertTrue(
                    all(rank % 8 == 0 for ranks in plan.values() for rank in ranks)
                )
                relative_error = abs(parameters - spec["dense_params_expected"]) / spec[
                    "dense_params_expected"
                ]
                self.assertLess(relative_error, 3e-4)

    @unittest.skipUnless(find_spec("torch"), "PyTorch is not installed")
    def test_257m_rank8_plan_matches_constructed_model(self):
        import torch

        from models.tucker_linear import TuckerLinear
        from models.utils import get_model
        from scripts.tucker_benchmark.benchmark_train_step import make_config

        args = self.args()
        args.variant = "static_tucker"
        args.microbatch = 1
        config = make_config(args)
        with torch.device("meta"):
            model = get_model(config)

        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            257_249_152,
        )
        modules = [module for module in model.modules() if isinstance(module, TuckerLinear)]
        self.assertEqual(len(modules), 84)
        self.assertTrue(
            all(
                value % 8 == 0
                for module in modules
                for value in (*module.modes, *module.ranks)
            )
        )

    @unittest.skipUnless(find_spec("torch"), "PyTorch is not installed")
    def test_dense_muon_step(self):
        import torch

        from models.utils import get_model
        from scripts.tucker_benchmark.benchmark_train_step import (
            build_dense_muon_optimizer,
            make_config,
        )

        args = self.args()
        args.variant = "dense_muon"
        args.microbatch = 2
        config = make_config(args)
        config.vocab_size = 32
        config.sequence_length = 4
        config.n_layer = 1
        config.n_embd = 8
        config.n_head = 2
        config.multiple_of = 4
        config.ffn_hidden_size = 16

        model = get_model(config)
        optimizer, split = build_dense_muon_optimizer(args, model)
        inputs = torch.randint(0, config.vocab_size, (2, config.sequence_length))
        targets = torch.randint(0, config.vocab_size, (2, config.sequence_length))
        loss = model(inputs, targets=targets)["loss"]
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(split["muon"], 0)
        self.assertGreater(split["adamw"], 0)

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
        config.tucker_rank_plan = None
        config.tucker_mode_multiple = 1
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
