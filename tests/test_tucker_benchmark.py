import unittest
from importlib.util import find_spec
from types import SimpleNamespace

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
            variant="tucker_parallel",
            sequence_length=1024,
            microbatch=1,
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
            tucker_cache_policy="recast",
            tucker_muon_core_microbatch=1,
            tucker_muon_streams=2,
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
        self.assertEqual(controls["tucker_forward_mode"], "chunked_contract")
        self.assertEqual(controls["storage_dtype"], "float32")
        self.assertEqual(controls["autocast_dtype"], "bfloat16")
        self.assertEqual(controls["tucker_rank_multiple"], 8)
        payload = {
            "status": "complete",
            "variant": {"name": "tucker_parallel"},
            "benchmark": controls,
            "samples": [{}] * 12,
        }
        self.assertTrue(result_matches_request(payload, "tucker_parallel", controls))
        changed = dict(controls, seed=1)
        self.assertFalse(result_matches_request(payload, "tucker_parallel", changed))

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
        from scripts.tucker_benchmark.benchmark_train_step import (
            instantiate_model,
            make_config,
        )

        args = self.args()
        args.variant = "tucker_reference"
        config = make_config(args)
        model = instantiate_model(config, torch.device("meta"))

        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            257_249_152,
        )
        modules = [module for module in model.modules() if isinstance(module, TuckerLinear)]
        self.assertEqual(len(modules), 84)
        self.assertIsInstance(model.lm_head, torch.nn.Linear)
        self.assertTrue(
            all(
                value % 8 == 0
                for module in modules
                for value in (*module.modes, *module.ranks)
            )
        )


if __name__ == "__main__":
    unittest.main()
