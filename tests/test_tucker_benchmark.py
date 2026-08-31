import unittest
from importlib.util import find_spec
from types import SimpleNamespace

from scripts.tucker_benchmark.common import (
    MODEL_SPECS,
    PROGRESSIVE_257M_STAGES,
    PROGRESSIVE_RANK_PROFILES,
    accumulation_steps,
    requested_controls,
    result_matches_request,
    summarize,
    tucker_benchmark_plan,
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
            tucker_rank_profile="iso",
            tucker_mode_layout="balanced4",
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

    def test_progressive_exact_and_rank8_profiles_match_stage_budgets(self):
        expected_exact = {
            "133m": 132_990_448,
            "160m": 159_965_968,
            "190m": 190_252_624,
            "225m": 224_635_072,
        }
        self.assertEqual(len(PROGRESSIVE_RANK_PROFILES), 8)
        for stage in PROGRESSIVE_257M_STAGES:
            with self.subTest(stage=stage["name"], alignment="exact"):
                geometry, plan, parameters, profile = tucker_benchmark_plan(
                    "257m", f"progressive_{stage['name']}_exact"
                )
                self.assertEqual(geometry["intermediate_size"], 2816)
                self.assertEqual(parameters, expected_exact[stage["name"]])
                self.assertEqual(profile["alignment"], "exact")
                self.assertTrue(
                    any(rank % 8 for ranks in plan.values() for rank in ranks)
                )
            with self.subTest(stage=stage["name"], alignment="rank8"):
                geometry, plan, parameters, profile = tucker_benchmark_plan(
                    "257m", f"progressive_{stage['name']}_rank8"
                )
                self.assertEqual(geometry["intermediate_size"], 2816)
                self.assertEqual(profile["alignment"], "rank8")
                self.assertTrue(
                    all(rank % 8 == 0 for ranks in plan.values() for rank in ranks)
                )
                relative_error = abs(parameters - stage["target_parameters"]) / stage[
                    "target_parameters"
                ]
                self.assertLess(relative_error, 3e-4)

    def test_order3_rank8_profiles_match_stage_budgets(self):
        targets = {"133m": 133_000_000, "225m": 225_000_000}
        for layout, singleton_index in (("order3_input", 3), ("order3_output", 0)):
            for stage, target in targets.items():
                with self.subTest(layout=layout, stage=stage):
                    _, plan, parameters, profile = tucker_benchmark_plan(
                        "257m", f"progressive_{stage}_rank8", layout
                    )
                    self.assertEqual(profile["mode_layout"], layout)
                    self.assertTrue(
                        all(ranks[singleton_index] == 1 for ranks in plan.values())
                    )
                    self.assertTrue(
                        all(
                            rank % 8 == 0
                            for ranks in plan.values()
                            for index, rank in enumerate(ranks)
                            if index != singleton_index
                        )
                    )
                    self.assertLess(abs(parameters - target) / target, 3e-4)

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
            257_155_584,
        )
        self.assertEqual(config.ffn_hidden_size, 3072)
        modules = [
            module for module in model.modules() if isinstance(module, TuckerLinear)
        ]
        self.assertEqual(len(modules), 84)
        self.assertIsInstance(model.lm_head, torch.nn.Linear)
        self.assertTrue(all(rank % 8 == 0 for module in modules for rank in module.ranks))
        self.assertTrue(all(max(module.modes) <= 64 for module in modules))

    @unittest.skipUnless(find_spec("torch"), "PyTorch is not installed")
    def test_order3_plan_matches_constructed_model(self):
        import torch

        from models.tucker_linear import TuckerLinear
        from scripts.tucker_benchmark.benchmark_train_step import (
            instantiate_model,
            make_config,
        )

        args = self.args()
        args.variant = "tucker_reference"
        args.tucker_rank_profile = "progressive_133m_rank8"
        args.tucker_mode_layout = "order3_input"
        config = make_config(args)
        model = instantiate_model(config, torch.device("meta"))
        modules = [module for module in model.modules() if isinstance(module, TuckerLinear)]

        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            config.target_parameter_count,
        )
        self.assertEqual(len(modules), 84)
        self.assertTrue(
            all(module.mode_layout == "order3_input" for module in modules)
        )
        self.assertTrue(all(len(module.active_factor_names) == 3 for module in modules))
        self.assertTrue(all(module.ranks[3] == 1 for module in modules))


if __name__ == "__main__":
    unittest.main()
