import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.monarch_benchmark.benchmark_train_step import (
    OPTIMIZER_BACKENDS,
    force_projector_resample,
    make_config,
    moment_state,
    projector_state,
    tensor_bytes,
    tensor_dtypes,
)
from scripts.monarch_benchmark.build_report import (
    comparison,
    load_results,
    render,
    result_key,
    validate_payload,
)
from scripts.monarch_benchmark.common import (
    COMMON_CONTROL_FIELDS,
    HARNESS_REVISION,
    MICROBATCHES,
    MODEL_SPECS,
    POINT_CONTROL_FIELDS,
    VARIANTS,
    accumulation_steps,
    model_geometry,
    parse_compute_apps,
    parse_gpu_inventory,
    percentile,
    result_is_complete,
    result_is_recorded,
    result_matches_request,
    summarize,
)
from scripts.monarch_benchmark.run_sweep import result_path
from models.llama import Llama
from models.monarch import apply_monarch


MONARCH_BLOCK_COUNTS = (4, 2)


def controls(microbatch=1, **overrides):
    values = {
        "harness_revision": HARNESS_REVISION,
        "storage_dtype": "bfloat16",
        "autocast_dtype": "bfloat16",
        "optimizer_moment_dtype": "bfloat16",
        "sequence_length": 1024,
        "microbatch": microbatch,
        "accumulation_steps": accumulation_steps(16384, microbatch, 1024),
        "tokens_per_step": 16384,
        "warmup_steps": 1,
        "measured_steps": 2,
        "compile_model": False,
        "monarch_blocks": 4,
        "density": 0.25,
        "update_proj_gap": 200,
        "resample_steps": 3,
        "lr": 1e-4,
        "momentum": 0.95,
        "betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "eps": 1e-7,
        "seed": 0,
        "contamination_poll_seconds": 0.25,
        "exclusive_gpu": False,
    }
    values.update(overrides)
    return values


def complete_result(model, variant, microbatch, median, *, state_bytes=8, resample_ms=None,
                    uuid="GPU-one", gpu_name="NVIDIA H100 80GB HBM3", exclusive=False):
    return {
        "status": "complete",
        "model": {
            "name": model["name"],
            "dense_equivalent_parameters": model["dense_params_expected"],
            "actual_parameters": model["dense_params_expected"],
        },
        "variant": {"name": variant},
        "benchmark": controls(microbatch, exclusive_gpu=exclusive),
        "gpu": {
            "uuid": uuid,
            "name": gpu_name,
            "foreign_processes_before": [],
            "foreign_processes_after": [],
            "contamination_monitor": {"foreign_processes_seen": [], "error": None},
        },
        "memory": {
            "peak_allocated_bytes": 100,
            "optimizer_state_bytes": state_bytes,
            "optimizer_moment_bytes": state_bytes,
            "optimizer_projector_bytes": 0,
        },
        "summary": {
            "host_total_ms": {"median": median, "p10": median, "p90": median},
            "optimizer_ms": {"median": 1.0},
            "tokens_per_second": {"median": 1000.0},
        },
        "resample_summary": (
            {"host_total_ms": {"median": resample_ms}} if resample_ms is not None else None
        ),
        "resample_memory": {"peak_allocated_bytes": 200} if resample_ms is not None else None,
        "samples": [{}, {}],
    }


def oom_result(model, variant, microbatch):
    return {
        "status": "oom",
        "model_size": model["name"],
        "variant": variant,
        "gpu_uuid": "GPU-one",
        "requested_controls": controls(microbatch),
    }


class BenchmarkMatrixTest(unittest.TestCase):
    def test_model_ramp_and_verified_parameter_counts(self):
        dense = [spec["dense_params_expected"] for spec in MODEL_SPECS]
        self.assertEqual(dense, sorted(dense))
        self.assertGreater(dense[-1], 6_800_000_000)
        self.assertLess(dense[-1], 7_000_000_000)
        for blocks in MONARCH_BLOCK_COUNTS:
            with self.subTest(blocks=blocks):
                monarch = [spec["monarch"][blocks]["params_expected"] for spec in MODEL_SPECS]
                self.assertEqual(monarch, sorted(monarch))
                self.assertTrue(all(m < d for m, d in zip(monarch, dense)))
        # Halving the block count doubles every factor, so two blocks always
        # carry more of the dense matrix than four do.
        for spec in MODEL_SPECS:
            self.assertGreater(
                spec["monarch"][2]["params_expected"], spec["monarch"][4]["params_expected"]
            )

    def test_iso_geometry_matches_the_dense_parameter_budget(self):
        for spec in MODEL_SPECS:
            for blocks in MONARCH_BLOCK_COUNTS:
                with self.subTest(model=spec["name"], blocks=blocks):
                    iso = model_geometry(spec, "monarch_muon_iso", blocks)
                    self.assertLess(
                        abs(iso["params_expected"] - spec["dense_params_expected"])
                        / spec["dense_params_expected"],
                        0.01,
                    )
                    self.assertEqual(iso["n_layer"], spec["n_layer"])
                    self.assertEqual(iso["n_embd"] // iso["n_head"], 128)
                    self.assertGreater(iso["n_embd"], spec["n_embd"])
        # Two blocks start closer to the dense budget, so they need less widening.
        for spec in MODEL_SPECS:
            self.assertLess(
                spec["monarch"][2]["iso"]["n_embd"], spec["monarch"][4]["iso"]["n_embd"]
            )

    def test_meta_models_match_declared_parameter_counts(self):
        with torch.device("meta"):
            for spec in MODEL_SPECS:
                config = make_config(spec, sequence_length=256)
                with self.subTest(model=spec["name"]):
                    dense = Llama(config)
                    self.assertEqual(
                        sum(parameter.numel() for parameter in dense.parameters()),
                        spec["dense_params_expected"],
                    )
                for blocks in MONARCH_BLOCK_COUNTS:
                    with self.subTest(model=spec["name"], blocks=blocks):
                        monarch = Llama(config)
                        with contextlib.redirect_stdout(io.StringIO()):
                            apply_monarch(monarch, nblocks=blocks, verbose=False)
                        self.assertEqual(
                            sum(parameter.numel() for parameter in monarch.parameters()),
                            spec["monarch"][blocks]["params_expected"],
                        )
                        iso_geometry = model_geometry(spec, "monarch_muon_iso", blocks)
                        iso = Llama(make_config(iso_geometry, sequence_length=256))
                        with contextlib.redirect_stdout(io.StringIO()):
                            apply_monarch(iso, nblocks=blocks, verbose=False)
                        self.assertEqual(
                            sum(parameter.numel() for parameter in iso.parameters()),
                            iso_geometry["params_expected"],
                        )

    def test_every_variant_has_a_declared_backend(self):
        self.assertEqual(
            sorted(variant["name"] for variant in VARIANTS), sorted(OPTIMIZER_BACKENDS)
        )
        families = {variant["family"] for variant in VARIANTS}
        self.assertTrue(families.issubset({"dense", "monarch", "memory_efficient"}))

    def test_microbatch_axis_keeps_tokens_per_step_fixed(self):
        for microbatch in MICROBATCHES:
            with self.subTest(microbatch=microbatch):
                accumulation = accumulation_steps(16384, microbatch, 1024)
                self.assertEqual(microbatch * 1024 * accumulation, 16384)
        with self.assertRaises(ValueError):
            accumulation_steps(16384, 3, 1024)

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


class ResumeTest(unittest.TestCase):
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

    def test_out_of_memory_counts_as_a_recorded_point(self):
        payload = oom_result(MODEL_SPECS[-1], "dense_adamw", 16)
        self.assertFalse(result_is_complete(payload))
        self.assertTrue(result_is_recorded(payload, gpu_uuid="GPU-one"))
        self.assertTrue(
            result_matches_request(
                payload,
                gpu_uuid="GPU-one",
                model_name=MODEL_SPECS[-1]["name"],
                variant_name="dense_adamw",
                controls=controls(16),
            )
        )
        self.assertFalse(
            result_matches_request(
                payload,
                gpu_uuid="GPU-one",
                model_name=MODEL_SPECS[-1]["name"],
                variant_name="dense_adamw",
                controls=controls(8),
            )
        )

    def test_resume_requires_exact_identity_and_controls(self):
        payload = complete_result(MODEL_SPECS[0], "dense_adamw", 4, 1.0)
        kwargs = {
            "gpu_uuid": "GPU-one",
            "model_name": MODEL_SPECS[0]["name"],
            "variant_name": "dense_adamw",
            "controls": controls(4),
        }
        self.assertTrue(result_matches_request(payload, **kwargs))
        payload["benchmark"]["sequence_length"] = 128
        self.assertFalse(result_matches_request(payload, **kwargs))

    def test_batch_size_is_part_of_the_result_identity(self):
        paths = {
            result_path(Path("/out"), "257m", "galore", microbatch) for microbatch in MICROBATCHES
        }
        self.assertEqual(len(paths), len(MICROBATCHES))
        self.assertIn("bs4", str(result_path(Path("/out"), "257m", "galore", 4)))

    def test_microbatch_is_a_point_control_not_a_sweep_control(self):
        self.assertNotIn("microbatch", COMMON_CONTROL_FIELDS)
        self.assertIn("microbatch", POINT_CONTROL_FIELDS)
        self.assertIn("tokens_per_step", COMMON_CONTROL_FIELDS)


class ProjectorAccountingTest(unittest.TestCase):
    """GaLore/APOLLO/Fira keep the projection in an attribute of a plain object,
    so a walk over tensors and containers alone reports it as zero bytes."""

    class Projector:
        def __init__(self, matrix):
            self.ortho_matrix = matrix
            self.update_proj_gap = 200
            self.rank = 4

    def optimizer_with(self, extra):
        parameter = torch.zeros(4, 4, dtype=torch.bfloat16)
        state = {parameter: {"step": 3, "exp_avg": torch.zeros(4, 4, dtype=torch.bfloat16), **extra}}
        return SimpleNamespace(state=state, param_groups=[{"update_proj_gap": 200, "params": [parameter]}])

    def test_projection_matrix_is_counted(self):
        matrix = torch.zeros(4, 8, dtype=torch.bfloat16)
        optimizer = self.optimizer_with({"projector": self.Projector(matrix)})
        moments = tensor_bytes(moment_state(optimizer))
        self.assertEqual(moments, 4 * 4 * 2)
        self.assertEqual(tensor_bytes(projector_state(optimizer)), 4 * 8 * 2)
        self.assertEqual(tensor_bytes(optimizer.state), moments + 4 * 8 * 2)

    def test_projector_dtypes_stay_out_of_the_moment_assertion(self):
        class IndexProjector:
            def __init__(self):
                self.indices = torch.zeros(6, dtype=torch.int64)

        optimizer = self.optimizer_with({"projector": IndexProjector()})
        self.assertIn("int64", tensor_dtypes(optimizer.state))
        self.assertEqual(
            sorted(tensor_dtypes(moment_state(optimizer), nonscalar_only=True)), ["bfloat16"]
        )

    def test_scalars_are_excluded_from_the_moment_assertion(self):
        optimizer = self.optimizer_with({"scaling_grad": torch.tensor(1.5, dtype=torch.bfloat16)})
        self.assertEqual(
            sorted(tensor_dtypes(moment_state(optimizer), nonscalar_only=True)), ["bfloat16"]
        )

    def test_forcing_a_resample_reaches_group_and_projector(self):
        optimizer = self.optimizer_with({"projector": self.Projector(torch.zeros(2, 2))})
        force_projector_resample(optimizer)
        self.assertEqual(optimizer.param_groups[0]["update_proj_gap"], 1)
        self.assertEqual(projector_state(optimizer)[0].update_proj_gap, 1)


class ReportTest(unittest.TestCase):
    def matrix(self):
        results = []
        for model in MODEL_SPECS[:2]:
            for microbatch in (1, 2):
                for index, variant in enumerate(("dense_adamw", "galore", "fira")):
                    results.append(
                        complete_result(
                            model,
                            variant,
                            microbatch,
                            median=float(index + 1),
                            state_bytes=8 // (index + 1),
                            resample_ms=None if variant == "dense_adamw" else 100.0,
                        )
                    )
        return results

    def test_report_accepts_a_two_dimensional_matrix(self):
        validated = validate_payload({"results": self.matrix()})
        compared = comparison(validated)
        self.assertEqual([model["name"] for model in compared["models"]],
                         [MODEL_SPECS[0]["name"], MODEL_SPECS[1]["name"]])
        self.assertEqual(compared["microbatches"], [1, 2])
        row = compared["models"][0]["rows"][0]
        self.assertAlmostEqual(row["cells"]["galore"]["speedup_vs_baseline"], 0.5)
        self.assertAlmostEqual(row["cells"]["galore"]["state_memory_ratio"], 0.5)
        # steady 2 ms, rebuild 100 ms, one rebuild every 200 steps
        self.assertAlmostEqual(row["cells"]["galore"]["amortized_ms"], 2.0 + 98.0 / 200)
        self.assertIsNone(row["cells"]["dense_adamw"]["resample_ms"])
        # dense AdamW never rebuilds, so it is charged nothing
        self.assertAlmostEqual(row["cells"]["dense_adamw"]["amortized_ms"], 1.0)

    def test_rebuild_cost_carries_across_the_batch_size_axis(self):
        # Measured only at microbatch 1, charged at every microbatch, because a
        # rebuild factorizes the gradient rather than the batch.
        results = [
            complete_result(MODEL_SPECS[0], "dense_adamw", 1, 1.0),
            complete_result(MODEL_SPECS[0], "galore", 1, 2.0, resample_ms=100.0),
            complete_result(MODEL_SPECS[0], "dense_adamw", 2, 3.0),
            complete_result(MODEL_SPECS[0], "galore", 2, 4.0, resample_ms=None),
        ]
        compared = comparison(validate_payload({"results": results}))
        wide = compared["models"][0]["rows"][1]["cells"]["galore"]
        self.assertIsNone(wide["resample_ms"])
        self.assertAlmostEqual(wide["resample_extra_ms"], 98.0)
        self.assertEqual(wide["resample_measured_at_microbatch"], 1)
        self.assertAlmostEqual(wide["amortized_ms"], 4.0 + 98.0 / 200)

    def test_report_keeps_out_of_memory_points_as_results(self):
        results = self.matrix()
        results.append(oom_result(MODEL_SPECS[0], "dense_adamw", 4))
        validated = validate_payload({"results": results})
        compared = comparison(validated)
        cells = compared["models"][0]["rows"][-1]["cells"]
        self.assertEqual(cells["dense_adamw"]["status"], "oom")
        self.assertEqual(compared["capacity"][0]["fits"]["dense_adamw"], 2)

    def test_report_rejects_duplicate_matrix_entry(self):
        results = self.matrix()
        results.append(results[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_payload({"results": results})

    def test_report_rejects_mixed_controls(self):
        results = self.matrix()
        results[-1]["benchmark"]["update_proj_gap"] = 4
        with self.assertRaisesRegex(ValueError, "controls differ"):
            validate_payload({"results": results})

    def test_report_rejects_multiple_gpus(self):
        results = self.matrix()
        results[-1]["gpu"]["uuid"] = "GPU-two"
        with self.assertRaisesRegex(ValueError, "multiple GPUs"):
            validate_payload({"results": results})

    def test_exclusive_scheduling_downgrades_to_one_card_model(self):
        # A scheduled card cannot be pinned across jobs, so a resumed cloud
        # sweep legitimately spans several cards of the same model.
        results = [
            complete_result(MODEL_SPECS[0], "dense_adamw", 1, 1.0, uuid="GPU-a", exclusive=True),
            complete_result(MODEL_SPECS[0], "galore", 1, 2.0, uuid="GPU-b", exclusive=True),
        ]
        validated = validate_payload({"results": results})
        self.assertEqual(validated["gpu_uuids"], ["GPU-a", "GPU-b"])
        results[1]["gpu"]["name"] = "NVIDIA A100 80GB PCIe"
        with self.assertRaisesRegex(ValueError, "different GPU models"):
            validate_payload({"results": results})

    def test_shared_machine_still_requires_one_physical_card(self):
        results = [
            complete_result(MODEL_SPECS[0], "dense_adamw", 1, 1.0, uuid="GPU-a"),
            complete_result(MODEL_SPECS[0], "galore", 1, 2.0, uuid="GPU-b"),
        ]
        with self.assertRaisesRegex(ValueError, "multiple GPUs"):
            validate_payload({"results": results})

    def test_load_results_merges_files_and_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "sweep-a" / "runs"
            runs.mkdir(parents=True)
            first = complete_result(MODEL_SPECS[0], "dense_adamw", 1, 1.0)
            (runs / "257m-dense_adamw-bs1.json").write_text(json.dumps(first))
            second = complete_result(MODEL_SPECS[0], "galore", 1, 2.0)
            bundle = root / "results.json"
            bundle.write_text(json.dumps({"results": [second]}))
            merged = load_results([root / "sweep-a", bundle])
        self.assertEqual(
            sorted(result_key(result) for result in merged["results"]),
            [(MODEL_SPECS[0]["name"], "dense_adamw", 1), (MODEL_SPECS[0]["name"], "galore", 1)],
        )

    def test_result_key_reads_both_payload_shapes(self):
        self.assertEqual(
            result_key(complete_result(MODEL_SPECS[0], "galore", 8, 1.0)),
            (MODEL_SPECS[0]["name"], "galore", 8),
        )
        self.assertEqual(
            result_key(oom_result(MODEL_SPECS[0], "galore", 8)),
            (MODEL_SPECS[0]["name"], "galore", 8),
        )

    def test_rendered_report_is_self_contained(self):
        validated = validate_payload({"results": self.matrix()})
        html = render(
            {
                "generated_at": "now",
                "source": "/tmp/results.json",
                "gpu_name": "NVIDIA H100 80GB HBM3",
                "controls": {field: validated["complete"][0]["benchmark"][field] for field in COMMON_CONTROL_FIELDS},
                "comparison": comparison(validated),
                "headline": [["a", "b"]],
                "conclusions": ["c"],
                "method": ["m"],
            }
        )
        self.assertNotIn("__DATA__", html)
        self.assertNotIn("</script>", html.split("<script>")[1].rsplit("</script>", 1)[0])
        self.assertIn("Memory-efficient optimizer benchmark", html)


if __name__ == "__main__":
    unittest.main()
