import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.tucker_linear import (
    TuckerLinear,
    auto_tucker_ranks,
    balanced_factor_pair,
    balanced_factor_triple,
    paired_factor_triples,
    retract_tucker_modules_,
    replace_all_linears_with_tucker,
    qr_retract_with_transport,
    tucker_retract_,
    tucker_retract_with_transport_,
)
from optim.tensorion import TensorionOptimizer


class TuckerLinearTest(unittest.TestCase):
    def test_balanced_factor_pairs_are_exact(self):
        for value, expected in (
            (1024, (32, 32)),
            (2816, (44, 64)),
            (50304, (192, 262)),
            (1023, (31, 33)),
            (17, (1, 17)),
        ):
            with self.subTest(value=value):
                pair = balanced_factor_pair(value)
                self.assertEqual(pair, expected)
                self.assertEqual(pair[0] * pair[1], value)

    def test_paired_factor_triples_are_exact_and_balanced(self):
        self.assertEqual(balanced_factor_triple(1024), (8, 8, 16))
        self.assertEqual(balanced_factor_triple(2816), (11, 16, 16))
        cases = {
            (1024, 1024): ((8, 8, 16), (8, 16, 8), (64, 128, 128)),
            (1024, 2816): ((8, 8, 16), (16, 16, 11), (128, 128, 176)),
            (2816, 1024): ((11, 16, 16), (16, 8, 8), (176, 128, 128)),
        }
        for shape, expected in cases.items():
            with self.subTest(shape=shape):
                input_modes, output_modes = paired_factor_triples(*shape)
                paired = tuple(
                    input_mode * output_mode
                    for input_mode, output_mode in zip(input_modes, output_modes)
                )
                self.assertEqual((input_modes, output_modes, paired), expected)

    def test_auto_ranks_for_experiment_shapes(self):
        expected = {
            (1024, 1023): ((32, 32, 31, 32), 27_679),
            (1024, 1024): ((32, 32, 32, 31), 28_704),
            (1024, 2816): ((32, 32, 44, 63), 37_040),
            (2816, 1024): ((44, 63, 32, 32), 37_040),
            (1024, 50304): ((32, 32, 192, 261), 89_314),
        }
        for shape, (ranks, gap) in expected.items():
            with self.subTest(shape=shape):
                self.assertEqual(auto_tucker_ranks(*shape), ranks)
                n1, n2 = balanced_factor_pair(shape[0])
                m1, m2 = balanced_factor_pair(shape[1])
                count = (
                    n1 * ranks[0]
                    + n2 * ranks[1]
                    + m1 * ranks[2]
                    + m2 * ranks[3]
                    + ranks[0] * ranks[1] * ranks[2] * ranks[3]
                )
                self.assertEqual(shape[0] * shape[1] - count, gap)

    def test_equal_parameter_count_and_effective_weight(self):
        torch.manual_seed(11)
        module = TuckerLinear(
            12,
            18,
            rank=1,
            bias=True,
            equal_params=True,
            dtype=torch.float64,
        )
        with torch.no_grad():
            module.residual_matrix.normal_()
            module.residual_tail.normal_()
        self.assertEqual(
            sum(parameter.numel() for parameter in module.parameters()),
            12 * 18 + 18,
        )
        x = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
        output = module(x)
        effective = module.materialize_weight(dtype=torch.float64)
        expected = F.linear(x, effective, module.bias)
        torch.testing.assert_close(output, expected, rtol=1e-10, atol=1e-10)

        output.square().mean().backward()
        self.assertIsNotNone(module.residual_matrix.grad)
        self.assertGreater(module.residual_matrix.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(module.residual_tail.grad)
        self.assertGreater(module.residual_tail.grad.abs().sum().item(), 0.0)
        for parameter in (
            module.U1,
            module.U2,
            module.U3,
            module.U4,
            module.core_matrix,
        ):
            self.assertIsNotNone(parameter.grad)

    def test_tail_only_residual_matches_effective_weight(self):
        torch.manual_seed(17)
        module = TuckerLinear(
            8,
            16,
            rank="auto",
            bias=False,
            equal_params=True,
            dtype=torch.float64,
        )
        self.assertEqual(module._residual_full_columns, 0)
        self.assertGreater(module._residual_partial_rows, 0)
        with torch.no_grad():
            module.residual_tail.normal_()
        x = torch.randn(3, 8, dtype=torch.float64)
        torch.testing.assert_close(
            module(x),
            F.linear(x, module.materialize_weight(dtype=torch.float64)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_materialized_forward_matches_explicit_linear(self):
        torch.manual_seed(19)
        module = TuckerLinear(
            12,
            18,
            rank=1,
            bias=True,
            equal_params=True,
            forward_mode="materialize",
            dtype=torch.float64,
        )
        with torch.no_grad():
            module.residual_matrix.normal_()
            module.residual_tail.normal_()
        x = torch.randn(2, 4, 12, dtype=torch.float64)
        effective = module.materialize_weight(dtype=torch.float64)
        torch.testing.assert_close(
            module(x),
            F.linear(x, effective, module.bias),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_order3_layouts_have_three_trainable_factors(self):
        cases = (
            ("order3_input", (2, 3, 4, 1), "U4"),
            ("order3_output", (1, 4, 2, 3), "U1"),
            ("order3_paired", (2, 2, 2, 1), "U4"),
        )
        for layout, ranks, inactive_name in cases:
            with self.subTest(layout=layout):
                torch.manual_seed(18)
                module = TuckerLinear(
                    12,
                    18,
                    rank=ranks,
                    bias=False,
                    equal_params=False,
                    forward_mode="chunked_contract",
                    contract_chunk_size=32,
                    mode_layout=layout,
                    dtype=torch.float64,
                )
                factor_parameters = {
                    name for name, _ in module.named_parameters() if name.startswith("U")
                }
                self.assertEqual(factor_parameters, set(module.active_factor_names))
                self.assertIn(inactive_name, dict(module.named_buffers()))
                self.assertEqual(getattr(module, inactive_name).item(), 1.0)
                self.assertEqual(
                    sum(parameter.numel() for parameter in module.parameters()),
                    module.tucker_parameter_count,
                )

                x = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
                output = module(x)
                torch.testing.assert_close(
                    output,
                    F.linear(x, module.materialize_weight(dtype=torch.float64)),
                    rtol=1e-10,
                    atol=1e-10,
                )
                output.square().mean().backward()
                self.assertIsNotNone(module.core_matrix.grad)
                self.assertTrue(
                    all(
                        getattr(module, name).grad is not None
                        for name in module.active_factor_names
                    )
                )
                self.assertIsNone(getattr(module, inactive_name).grad)

                weight_before = module.materialize_weight(dtype=torch.float64)
                module.retract_()
                self.assertEqual(
                    retract_tucker_modules_(module)["factors"],
                    3,
                )
                torch.testing.assert_close(
                    module.materialize_weight(dtype=torch.float64),
                    weight_before,
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_paired_order3_custom_backward_matches_materialized(self):
        torch.manual_seed(20260901)
        custom = TuckerLinear(
            12,
            18,
            rank=(2, 3, 4, 1),
            bias=False,
            equal_params=False,
            forward_mode="chunked_contract",
            mode_layout="order3_paired",
            dtype=torch.float64,
        )
        reference = TuckerLinear(
            12,
            18,
            rank=(2, 3, 4, 1),
            bias=False,
            equal_params=False,
            forward_mode="materialize",
            mode_layout="order3_paired",
            dtype=torch.float64,
        )
        reference.load_state_dict(custom.state_dict())
        custom_input = torch.randn(7, 12, dtype=torch.float64, requires_grad=True)
        reference_input = custom_input.detach().clone().requires_grad_(True)
        grad_output = torch.randn(7, 18, dtype=torch.float64)

        custom_output = custom(custom_input)
        reference_output = reference(reference_input)
        custom_output.backward(grad_output)
        reference_output.backward(grad_output)

        torch.testing.assert_close(custom_output, reference_output)
        torch.testing.assert_close(custom_input.grad, reference_input.grad)
        for (custom_name, custom_parameter), (reference_name, reference_parameter) in zip(
            custom.named_parameters(), reference.named_parameters()
        ):
            self.assertEqual(custom_name, reference_name)
            torch.testing.assert_close(custom_parameter.grad, reference_parameter.grad)

    def test_paired_order3_persistent_work_cache_tracks_parameter_versions(self):
        from models.tucker_paired import paired_tucker_linear

        module = TuckerLinear(
            12,
            18,
            rank=(2, 3, 4, 1),
            bias=False,
            equal_params=False,
            forward_mode="chunked_contract",
            mode_layout="order3_paired",
            dtype=torch.float64,
        )
        x = torch.randn(5, 12, dtype=torch.float64)
        with torch.no_grad():
            first = paired_tucker_linear(x, module, cache_policy="persistent")
            first_cache = module._paired_tucker_work_cache
            second = paired_tucker_linear(x, module, cache_policy="persistent")
            self.assertIs(module._paired_tucker_work_cache, first_cache)
            torch.testing.assert_close(second, first)

            module.U1.add_(0.01)
            updated = paired_tucker_linear(x, module, cache_policy="persistent")
            self.assertNotEqual(module._paired_tucker_work_cache[0], first_cache[0])
            torch.testing.assert_close(
                updated,
                F.linear(x, module.materialize_weight(dtype=x.dtype)),
            )

    def test_materialized_weight_has_dense_shape(self):
        for in_features, out_features in (
            (12, 18),
            (1024, 1024),
            (1024, 2816),
        ):
            with self.subTest(
                in_features=in_features,
                out_features=out_features,
            ):
                module = TuckerLinear(
                    in_features,
                    out_features,
                    rank=1,
                    bias=False,
                    equal_params=False,
                    dtype=torch.float64,
                )
                self.assertEqual(
                    module.materialize_weight(dtype=torch.float64).shape,
                    (out_features, in_features),
                )

    def test_tucker_retraction_preserves_tensor_and_orthonormalizes(self):
        torch.manual_seed(23)
        core = torch.randn(2, 3, 2, 4, dtype=torch.float64)
        factors = [
            torch.randn(5, 2, dtype=torch.float64),
            torch.randn(6, 3, dtype=torch.float64),
            torch.randn(4, 2, dtype=torch.float64),
            torch.randn(7, 4, dtype=torch.float64),
        ]
        before = torch.einsum(
            "abcd,ia,jb,kc,ld->ijkl",
            core,
            *factors,
        )

        core = tucker_retract_(core, factors)
        after = torch.einsum(
            "abcd,ia,jb,kc,ld->ijkl",
            core,
            *factors,
        )
        torch.testing.assert_close(after, before, rtol=1e-10, atol=1e-10)
        for factor in factors:
            torch.testing.assert_close(
                factor.T @ factor,
                torch.eye(factor.shape[1], dtype=factor.dtype),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_qr_transport_matches_finite_difference_and_is_tangent(self):
        torch.manual_seed(24)
        factor = torch.randn(7, 3, dtype=torch.float64)
        tangent = torch.randn_like(factor)
        Q, R, dQ, dR = qr_retract_with_transport(factor, tangent)
        epsilon = 1e-7
        Q_eps, R_eps, _, _ = qr_retract_with_transport(
            factor + epsilon * tangent,
            None,
        )

        torch.testing.assert_close(
            Q_eps,
            Q + epsilon * dQ,
            rtol=2e-6,
            atol=2e-8,
        )
        torch.testing.assert_close(
            R_eps,
            R + epsilon * dR,
            rtol=2e-6,
            atol=2e-8,
        )
        torch.testing.assert_close(
            Q.mT @ dQ + dQ.mT @ Q,
            torch.zeros(3, 3, dtype=torch.float64),
            rtol=0.0,
            atol=1e-10,
        )

    def test_tucker_vector_transport_matches_full_gauge_map_differential(self):
        torch.manual_seed(25)
        core = torch.randn(2, 3, 2, 2, dtype=torch.float64)
        factors = [
            torch.randn(5, 2, dtype=torch.float64),
            torch.randn(6, 3, dtype=torch.float64),
            torch.randn(4, 2, dtype=torch.float64),
            torch.randn(7, 2, dtype=torch.float64),
        ]
        core_tangent = torch.randn_like(core)
        factor_tangents = [torch.randn_like(factor) for factor in factors]
        (
            retracted_core,
            retracted_factors,
            transported_core,
            transported_factors,
        ) = tucker_retract_with_transport_(
            core,
            factors,
            core_tangent,
            factor_tangents,
        )

        epsilon = 1e-7
        perturbed_core, perturbed_factors, _, _ = tucker_retract_with_transport_(
            core + epsilon * core_tangent,
            [
                factor + epsilon * tangent
                for factor, tangent in zip(factors, factor_tangents)
            ],
            None,
            [None] * 4,
        )
        torch.testing.assert_close(
            perturbed_core,
            retracted_core + epsilon * transported_core,
            rtol=4e-6,
            atol=5e-8,
        )
        for perturbed, retracted, transported in zip(
            perturbed_factors,
            retracted_factors,
            transported_factors,
        ):
            torch.testing.assert_close(
                perturbed,
                retracted + epsilon * transported,
                rtol=4e-6,
                atol=5e-8,
            )

    def test_module_retraction_preserves_effective_dense_weight(self):
        torch.manual_seed(29)
        module = TuckerLinear(
            12,
            18,
            rank=2,
            bias=False,
            equal_params=False,
            dtype=torch.float64,
        )
        with torch.no_grad():
            for factor in (module.U1, module.U2, module.U3, module.U4):
                factor.add_(0.2 * torch.randn_like(factor))

        before = module.materialize_weight(dtype=torch.float64)
        diagnostics = retract_tucker_modules_(
            module,
            compute_diagnostics=True,
        )
        after = module.materialize_weight(dtype=torch.float64)

        self.assertEqual(diagnostics["modules"], 1)
        self.assertEqual(diagnostics["factors"], 4)
        self.assertLess(diagnostics["max_orthogonality_error"], 1e-10)
        torch.testing.assert_close(after, before, rtol=1e-10, atol=1e-10)

    def test_optimizer_state_transport_integrates_with_retraction_hook(self):
        torch.manual_seed(30)
        module = TuckerLinear(
            12,
            18,
            rank=2,
            bias=False,
            equal_params=False,
            dtype=torch.float64,
        )
        r1, r2, r3, r4 = module.ranks
        optimizer = TensorionOptimizer(
            tensorion_params=[
                ("core", module.core_matrix, (r3, r4, r1, r2))
            ],
            muon_params=[
                (name, parameter)
                for name, parameter in (
                    ("U1", module.U1),
                    ("U2", module.U2),
                    ("U3", module.U3),
                    ("U4", module.U4),
                )
            ],
            adamw_param_groups=[],
            lr=1e-3,
            momentum=0.95,
            orthogonalization="svd",
        )
        for parameter in module.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        before = module.materialize_weight(dtype=torch.float64)

        diagnostics = retract_tucker_modules_(
            module,
            optimizer=optimizer,
            transport_optimizer_state=True,
            compute_diagnostics=True,
        )
        after = module.materialize_weight(dtype=torch.float64)

        self.assertEqual(diagnostics["transported_cores"], 1)
        self.assertEqual(diagnostics["transported_factors"], 4)
        self.assertLess(diagnostics["max_momentum_tangency_error"], 1e-10)
        torch.testing.assert_close(after, before, rtol=1e-10, atol=1e-10)

    def test_no_equal_params_is_actually_smaller(self):
        module = TuckerLinear(12, 18, rank=1, bias=False, equal_params=False)
        self.assertIsNone(module.residual_matrix)
        self.assertIsNone(module.residual_tail)
        self.assertLess(
            sum(parameter.numel() for parameter in module.parameters()),
            12 * 18,
        )

    def test_pure_tucker_allows_rank_count_above_one_dense_layer(self):
        module = TuckerLinear(
            12,
            18,
            rank=100,
            bias=False,
            equal_params=False,
        )
        self.assertEqual(module.ranks, module.modes)
        self.assertGreater(module.tucker_parameter_count, 12 * 18)
        self.assertEqual(module.residual_parameter_count, 0)
        self.assertIsNone(module.residual_matrix)
        self.assertIsNone(module.residual_tail)

    def test_rank_259_matches_control_within_requested_tolerance(self):
        shapes = (
            ((1024, 1023), 36, (32, 32, 31, 33), 1_051_650),
            ((1024, 1024), 12, (32, 32, 32, 32), 1_052_672),
            ((1024, 2816), 24, (32, 32, 44, 64), 2_891_664),
            ((2816, 1024), 12, (44, 64, 32, 32), 2_891_664),
            ((1024, 50304), 1, (32, 32, 192, 259), 51_028_242),
        )
        tucker_linear_parameters = 0
        for shape, count, expected_ranks, expected_parameters in shapes:
            modes = (*balanced_factor_pair(shape[0]), *balanced_factor_pair(shape[1]))
            ranks = tuple(min(259, mode) for mode in modes)
            self.assertEqual(ranks, expected_ranks)
            parameters = sum(
                mode * rank for mode, rank in zip(modes, ranks)
            ) + math.prod(ranks)
            self.assertEqual(parameters, expected_parameters)
            tucker_linear_parameters += count * parameters

        model_parameters = tucker_linear_parameters + 51_561_448
        self.assertEqual(model_parameters, 257_181_058)
        self.assertEqual(model_parameters - 257_188_864, -7_806)
        self.assertLessEqual(abs(model_parameters - 257_188_864), 12_312)

    def test_rank_259_standard_attention_matches_control_tolerance(self):
        full_1024_tucker = 1_052_672
        full_mlp_tucker = 2_891_664
        lm_head_tucker = 51_028_242
        nonlinear_parameters = 51_536_896

        model_parameters = (
            48 * full_1024_tucker
            + 36 * full_mlp_tucker
            + lm_head_tucker
            + nonlinear_parameters
        )
        self.assertEqual(model_parameters, 257_193_298)
        self.assertEqual(model_parameters - 257_188_864, 4_434)
        self.assertLessEqual(abs(model_parameters - 257_188_864), 12_312)

    def test_small_positive_control_budget_is_exact_and_trainable(self):
        module = TuckerLinear(
            12,
            18,
            rank=1,
            bias=False,
            equal_params=True,
            extra_parameters=7,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in module.parameters()),
            12 * 18 + 7,
        )
        self.assertEqual(
            module.residual_parameter_count,
            12 * 18 + 7 - module.tucker_parameter_count,
        )

    def test_invalid_manual_rank_reports_dense_budget(self):
        with self.assertRaisesRegex(ValueError, "exceeding dense budget"):
            TuckerLinear(12, 18, rank=6, bias=False, equal_params=True)

    def test_llama_replacement_covers_every_linear_and_preserves_count(self):
        from models.llama import Llama

        config = SimpleNamespace(
            model="llama",
            vocab_size=32,
            sequence_length=4,
            n_embd=8,
            n_head=1,
            n_layer=1,
            dropout=0.0,
            bias=False,
            init_std=0.02,
            rmsnorm_eps=1e-5,
            multiple_of=4,
            ffn_hidden_size=16,
            label_smoothing=0.0,
            qkv_clipping=False,
            qkv_clipping_factor=1.0,
            attention_type="tensorized",
            tensorized_mode="reconstruction",
            tensorized_rank=3,
            tensorized_num_cores=2,
            tensorized_causal=True,
            tensorized_query_chunk_size=2,
            linear_parameterization="tucker",
            tucker_rank="auto",
            tucker_ranks=None,
            tucker_equal_params=True,
            tucker_forward_mode="chunked_contract",
            tucker_contract_chunk_size=17,
            tucker_head_contract_chunk_size=5,
            fp8=False,
        )
        model = Llama(config)
        before = sum(parameter.numel() for parameter in model.parameters())
        stats = replace_all_linears_with_tucker(model, config)
        after = sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(before, after)
        self.assertEqual(stats.parameters_before, stats.parameters_after)
        self.assertFalse(any(isinstance(module, nn.Linear) for module in model.modules()))
        self.assertIsInstance(model.lm_head, TuckerLinear)
        self.assertEqual(model.lm_head.contract_chunk_size, 5)
        block = model.transformer.h[0]
        for projection in (
            block.attn.q_proj,
            block.attn.k_proj,
            block.attn.v_proj,
            block.attn.o_proj,
            block.mlp.gate_proj,
            block.mlp.up_proj,
            block.mlp.down_proj,
        ):
            self.assertIsInstance(projection, TuckerLinear)
            self.assertEqual(projection.contract_chunk_size, 17)

        tokens = torch.randint(0, config.vocab_size, (2, config.sequence_length))
        result = model(tokens, targets=tokens, get_logits=True)
        self.assertEqual(result["logits"].shape, (2, 4, 32))
        result["loss"].backward()

        grouped = model.get_parameter_group_specs()
        grouped_names = set().union(*(set(group["params"]) for group in grouped))
        self.assertEqual(grouped_names, set(dict(model.named_parameters())))
        for name, parameter in model.named_parameters():
            if "lm_head" not in name and name.endswith("residual_matrix"):
                self.assertEqual(parameter.ndim, 2)
            if name.endswith("residual_tail"):
                self.assertEqual(parameter.ndim, 1)

    def test_dense_adamw_matrix_mode_keeps_lm_head_dense(self):
        from models.llama import Llama

        config = SimpleNamespace(
            model="llama",
            vocab_size=32,
            sequence_length=4,
            batch_size=2,
            n_embd=8,
            n_head=1,
            n_layer=1,
            dropout=0.0,
            bias=False,
            init_std=0.02,
            rmsnorm_eps=1e-5,
            multiple_of=4,
            ffn_hidden_size=16,
            label_smoothing=0.0,
            qkv_clipping=False,
            qkv_clipping_factor=1.0,
            attention_type="standard",
            linear_parameterization="tucker",
            tucker_rank="auto",
            tucker_ranks=None,
            tucker_equal_params=True,
            tucker_forward_mode="auto",
            tucker_dense_adamw_matrices=True,
            target_parameter_count=0,
            target_parameter_tolerance=0,
            fp8=False,
            fp8_optim=False,
        )
        model = Llama(config)
        before = sum(parameter.numel() for parameter in model.parameters())
        stats = replace_all_linears_with_tucker(model, config)

        self.assertIsInstance(model.lm_head, nn.Linear)
        self.assertIsInstance(model.transformer.h[0].attn.q_proj, TuckerLinear)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            before,
        )
        self.assertEqual(stats.modules, 7)

    def test_adaptive_flops_differ_from_parameter_count_formula(self):
        module = TuckerLinear(12, 18, rank=1, bias=False, equal_params=True)
        self.assertNotEqual(
            module.forward_flops_per_token,
            2 * module.weight_parameter_count,
        )

    def test_model_level_positive_parameter_target_is_exact(self):
        from models.llama import Llama

        config = SimpleNamespace(
            model="llama",
            vocab_size=32,
            sequence_length=4,
            batch_size=2,
            n_embd=8,
            n_head=1,
            n_layer=1,
            dropout=0.0,
            bias=False,
            init_std=0.02,
            rmsnorm_eps=1e-5,
            multiple_of=4,
            ffn_hidden_size=16,
            label_smoothing=0.0,
            qkv_clipping=False,
            qkv_clipping_factor=1.0,
            attention_type="tensorized",
            tensorized_mode="reconstruction",
            tensorized_rank=3,
            tensorized_num_cores=2,
            tensorized_causal=True,
            tensorized_query_chunk_size=2,
            linear_parameterization="tucker",
            tucker_rank="auto",
            tucker_ranks=None,
            tucker_equal_params=True,
            tucker_forward_mode="auto",
            target_parameter_count=0,
            fp8=False,
            fp8_optim=False,
        )
        model = Llama(config)
        dense_count = sum(parameter.numel() for parameter in model.parameters())
        config.target_parameter_count = dense_count + 7
        stats = replace_all_linears_with_tucker(model, config)
        self.assertEqual(stats.parameters_after, dense_count + 7)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            dense_count + 7,
        )

        # In pure mode the target is validation-only: no correction is added.
        config.tucker_equal_params = False
        config.tucker_rank = "100"
        config.target_parameter_count = 0
        config.target_parameter_tolerance = 0
        pure_model = Llama(config)
        pure_stats = replace_all_linears_with_tucker(pure_model, config)
        self.assertEqual(pure_stats.residual_parameters, 0)

        config.target_parameter_count = pure_stats.parameters_after + 7
        config.target_parameter_tolerance = 7
        validated_model = Llama(config)
        validated_stats = replace_all_linears_with_tucker(validated_model, config)
        self.assertEqual(validated_stats.parameter_difference_from_target, -7)
        self.assertEqual(validated_stats.residual_parameters, 0)
        self.assertEqual(
            validated_stats.parameters_after,
            pure_stats.parameters_after,
        )

        config.target_parameter_tolerance = 6
        rejected_model = Llama(config)
        with self.assertRaisesRegex(ValueError, "rank-only parameter check failed"):
            replace_all_linears_with_tucker(rejected_model, config)

        config.target_parameter_count = -1
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            replace_all_linears_with_tucker(Llama(config), config)
        config.target_parameter_count = 0
        config.target_parameter_tolerance = -1
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            replace_all_linears_with_tucker(Llama(config), config)


if __name__ == "__main__":
    unittest.main()
