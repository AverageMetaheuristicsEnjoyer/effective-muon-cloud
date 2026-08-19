import contextlib
import io
import unittest

import torch

from scripts.monarch_benchmark.benchmark_train_step import make_config
from scripts.monarch_benchmark.build_report import comparison, validate_payload

from scripts.monarch_benchmark.common import (
    COMMON_CONTROL_FIELDS,
    HARNESS_REVISION,
    MODEL_SPECS,
    parse_compute_apps,
    parse_gpu_inventory,
    percentile,
    result_is_complete,
    result_matches_request,
    summarize,
)
from models.llama import Llama
from models.monarch import apply_monarch


class MonarchLargeBenchmarkTest(unittest.TestCase):
    def test_model_ramp_and_verified_parameter_counts(self):
        dense = [spec["dense_params_expected"] for spec in MODEL_SPECS]
        monarch = [spec["monarch_params_expected"] for spec in MODEL_SPECS]
        self.assertEqual(dense, sorted(dense))
        self.assertEqual(monarch, sorted(monarch))
        self.assertGreater(dense[-1], 6_800_000_000)
        self.assertLess(dense[-1], 7_000_000_000)
        self.assertTrue(all(m < d for m, d in zip(monarch, dense)))

    def test_meta_models_match_declared_parameter_counts(self):
        with torch.device("meta"):
            for spec in MODEL_SPECS:
                with self.subTest(model=spec["name"]):
                    config = make_config(spec, sequence_length=256)
                    dense = Llama(config)
                    self.assertEqual(
                        sum(parameter.numel() for parameter in dense.parameters()),
                        spec["dense_params_expected"],
                    )
                    monarch = Llama(config)
                    with contextlib.redirect_stdout(io.StringIO()):
                        apply_monarch(monarch, nblocks=4, verbose=False)
                    self.assertEqual(
                        sum(parameter.numel() for parameter in monarch.parameters()),
                        spec["monarch_params_expected"],
                    )

    def test_statistics_are_deterministic(self):
        values = [1.0, 2.0, 3.0, 4.0]
        stats = summarize(values)
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["median"], 2.5)
        self.assertAlmostEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(percentile(values, 0.1), 1.3)
        self.assertAlmostEqual(percentile(values, 0.9), 3.7)

    def test_nvidia_smi_parsers(self):
        gpus = parse_gpu_inventory(
            "2, NVIDIA A100 80GB PCIe, GPU-abc, 81920, 7, 0, 34, P0\n"
        )
        self.assertEqual(gpus[0]["uuid"], "GPU-abc")
        self.assertEqual(gpus[0]["memory_used_mb"], 7)
        processes = parse_compute_apps(
            "GPU-abc, 1234, python, 8192\nGPU-def, 99, python3, [Not Supported]\n"
        )
        self.assertEqual(processes[0]["pid"], 1234)
        self.assertIsNone(processes[1]["used_memory_mb"])

    def test_complete_result_requires_all_samples(self):
        payload = {
            "status": "complete",
            "gpu": {"uuid": "GPU-abc"},
            "benchmark": {"measured_steps": 2},
            "samples": [{}, {}],
        }
        self.assertTrue(result_is_complete(payload, gpu_uuid="GPU-abc"))
        self.assertFalse(result_is_complete(payload, gpu_uuid="GPU-other"))
        payload["samples"].pop()
        self.assertFalse(result_is_complete(payload))

    def test_resume_requires_exact_identity_and_controls(self):
        controls = {
            "harness_revision": HARNESS_REVISION,
            "sequence_length": 256,
            "measured_steps": 2,
        }
        payload = {
            "status": "complete",
            "gpu": {"uuid": "GPU-abc"},
            "model": {"name": "257m"},
            "variant": {"name": "dense_adamw"},
            "benchmark": dict(controls),
            "samples": [{}, {}],
        }
        kwargs = {
            "gpu_uuid": "GPU-abc",
            "model_name": "257m",
            "variant_name": "dense_adamw",
            "controls": controls,
        }
        self.assertTrue(result_matches_request(payload, **kwargs))
        payload["benchmark"]["sequence_length"] = 128
        self.assertFalse(result_matches_request(payload, **kwargs))

    def test_report_requires_complete_single_gpu_matrix(self):
        results = []
        for model in MODEL_SPECS:
            for variant_index, variant in enumerate(
                ("monarch_muon", "dense_adamw", "dense_muon")
            ):
                median = float(variant_index + 1)
                results.append(
                    {
                        "status": "complete",
                        "gpu": {"uuid": "GPU-one"},
                        "model": {
                            "name": model["name"],
                            "dense_equivalent_parameters": model[
                                "dense_params_expected"
                            ],
                            "actual_parameters": model["dense_params_expected"],
                        },
                        "variant": {"name": variant},
                        "benchmark": {
                            "harness_revision": HARNESS_REVISION,
                            "autocast_dtype": "bfloat16",
                            "optimizer_moment_dtype": "bfloat16",
                            "sequence_length": 256,
                            "microbatch": 1,
                            "accumulation_steps": 4,
                            "tokens_per_step": 1024,
                            "storage_dtype": "bfloat16",
                            "warmup_steps": 1,
                            "measured_steps": 2,
                            "compile_model": False,
                            "monarch_blocks": 4,
                            "lr": 1e-4,
                            "momentum": 0.95,
                            "betas": [0.9, 0.95],
                            "weight_decay": 0.1,
                            "eps": 1e-7,
                            "seed": 0,
                            "contamination_poll_seconds": 0.25,
                        },
                        "gpu": {
                            "uuid": "GPU-one",
                            "foreign_processes_before": [],
                            "foreign_processes_after": [],
                            "contamination_monitor": {
                                "foreign_processes_seen": [],
                                "error": None,
                            },
                        },
                        "summary": {"host_total_ms": {"median": median}},
                        "samples": [{}, {}],
                    }
                )
        validated = validate_payload({"results": results})
        compared = comparison(validated)
        self.assertEqual(len(compared["rows"]), len(MODEL_SPECS))
        self.assertEqual(compared["largest"]["speedup_vs_adamw"], 2.0)
        results.pop()
        with self.assertRaises(ValueError):
            validate_payload({"results": results})

    def test_report_rejects_duplicate_matrix_entry(self):
        controls = {
            field: {
                "harness_revision": HARNESS_REVISION,
                "storage_dtype": "bfloat16",
                "autocast_dtype": "bfloat16",
                "optimizer_moment_dtype": "bfloat16",
                "sequence_length": 256,
                "microbatch": 1,
                "accumulation_steps": 4,
                "tokens_per_step": 1024,
                "warmup_steps": 1,
                "measured_steps": 1,
                "compile_model": False,
                "monarch_blocks": 4,
                "lr": 1e-4,
                "momentum": 0.95,
                "betas": [0.9, 0.95],
                "weight_decay": 0.1,
                "eps": 1e-7,
                "seed": 0,
                "contamination_poll_seconds": 0.25,
            }[field]
            for field in COMMON_CONTROL_FIELDS
        }
        result = {
            "status": "complete",
            "gpu": {
                "uuid": "GPU-one",
                "foreign_processes_before": [],
                "foreign_processes_after": [],
                "contamination_monitor": {"foreign_processes_seen": [], "error": None},
            },
            "model": {"name": "257m"},
            "variant": {"name": "monarch_muon"},
            "benchmark": controls,
            "samples": [{}],
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_payload({"results": [result] * 15})


if __name__ == "__main__":
    unittest.main()
