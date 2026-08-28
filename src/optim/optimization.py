import torch
from .memory_efficient.frugal import AdamW, GaloreAdamW, CoordAdamW, BlockAdamW, SGD, GaloreSGD, CoordSGD, BlockSGD, Lion, GaloreLion, CoordLion, BlockLion, Adalayer, GaloreAdalayer, CoordAdalayer, BlockAdalayer
from .memory_efficient.fira import FiraAdamW
from .memory_efficient.galore import GaLoreAdafactor, AdaMeM
from .memory_efficient.apollo import APOLLOAdamW
from .memory_efficient.ldadam import LDAdamW
from .memory_efficient.coap import COAPAdamW
from .memory_efficient.cosmos import COSMOS
from .memory_efficient.sumo import SUMO
from .memory_efficient.badam import BlockOptimizer, BlockOptimizerRatio
from .memory_efficient.adam_mini import Adam_mini
from .memory_efficient.slim_adam import SlimAdamW
from .memory_efficient.slim_adam import DEFAULT_LAYER_MAP_PATH as SLIM_ADAM_DEFAULT_LAYER_MAP
from .memory_efficient.lora import LoRAOptimizer
from .memory_efficient.lora_rite import LoRARiteOptimizer
from .memory_efficient.loro import LOROAdamW
from .sota_opt import AdEMAMix, dion, Adan, ADOPT, SOAP, MARS, MARS_M, SWAN
from distributed_shampoo import DistributedShampoo


def get_optimizer(param_groups, args, model=None, qargs=None):
    
    optimizer_name = args.opt.lower()
    assert optimizer_name != "badam" or model is not None, "BAdam requires model for the initialization."
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)#, foreach=False, fused=False)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)#, foreach=False, fused=False)
    elif optimizer_name == "ademamix":
        optimizer = AdEMAMix(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2, args.ademamix_beta3),
            alpha=args.ademamix_alpha,
            beta3_warmup=args.ademamix_beta3_warmup_steps,
            alpha_warmup=args.ademamix_alpha_warmup_steps,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    elif optimizer_name == "muon":
        for group in param_groups:
            if not group.get("is_proj_params", False):
                if args.muon_1d_backup == "adan":
                    group["algorithm"] = "adan"
                else:
                    group["algorithm"] = "adamw"
        optimizer = dion.Muon(
            param_groups,
            distributed_mesh=model.process_group,
            lr=args.lr,
            mu=args.momentum,
            betas=(args.beta1, args.beta2),
            adan_betas=(args.adan_beta1, args.adan_beta2, args.adan_beta3),
            weight_decay=args.weight_decay,
            epsilon=args.eps,
            nesterov=args.nesterov,
            adjust_lr=args.muon_adjust_lr,
            newton_schulz_func_name=args.newton_schulz_func,
            pre_orth_update_name=args.muon_pre_orth_update,
            muon_ns_steps=args.muon_ns_steps,
            headwise=args.muon_headwise,
            adamw_lr_scale=args.muon_adamw_lr_scale,
            dampening=args.dampening
        )
    elif optimizer_name == "normuon":
        for group in param_groups:
            if not group.get("is_proj_params", False):
                group["algorithm"] = "adamw"
        optimizer = dion.NorMuon(
            param_groups,
            distributed_mesh=model.process_group,
            lr=args.lr,
            mu=args.momentum,
            muon_beta2=args.muon_beta2,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            epsilon=args.eps,
            nesterov=args.nesterov,
            adjust_lr=args.muon_adjust_lr,
            use_adamuon=False,
        )
    elif optimizer_name == "adamuon":
        for group in param_groups:
            if not group.get("is_proj_params", False):
                group["algorithm"] = "adamw"
        optimizer = dion.NorMuon(
            param_groups,
            distributed_mesh=model.process_group,
            lr=args.lr,
            mu=args.momentum,
            muon_beta2=args.muon_beta2,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            epsilon=args.eps,
            nesterov=args.nesterov,
            adjust_lr=args.muon_adjust_lr,
            use_adamuon=True,
        )
    elif optimizer_name == "adan":
        optimizer = Adan.Adan(
            param_groups,
            lr=args.lr,
            betas=(args.adan_beta1, args.adan_beta2, args.adan_beta3),
            eps=args.eps,
            weight_decay=args.weight_decay,
            max_grad_norm=args.adan_max_grad_norm,
            no_prox=args.adan_no_prox,
            # foreach = False,
            # fused = False,
        )
    elif optimizer_name == "nadam":
        optimizer = torch.optim.NAdam(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            momentum_decay=args.nadam_momentum_decay,
            decoupled_weight_decay=True,
            # foreach = False,
            # fused = False,
        )
    elif optimizer_name == "adopt":
        optimizer = ADOPT(
            param_groups,
            lr=args.lr,
            betas=(args.adopt_beta1, args.adopt_beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            decouple=args.adopt_decouple
        )
    elif optimizer_name == "shampoo":
        optimizer = DistributedShampoo(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            beta3=args.shampoo_beta3,
            epsilon=args.eps,
            weight_decay=args.weight_decay,
            precondition_frequency=args.shampoo_preconditioner_frequency,
            max_preconditioner_dim=args.shampoo_max_preconditioner_dim,
        )
    elif optimizer_name == "soap":
        optimizer = SOAP(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            shampoo_beta=args.shampoo_beta,
            eps=args.eps,
            weight_decay=args.weight_decay,
            precondition_frequency=args.update_gap,
            precondition_embed_debed=args.soap_precondition_embed_debed,
        )
    elif optimizer_name == "mars":
        optimizer = MARS(
            param_groups,
            lr=args.lr,
            betas=(args.mars_beta1, args.mars_beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            # amsgrad=False,
            gamma=args.mars_gamma,
            # is_approx=True,
            mars_type=args.mars_type,

            # optimize_1d=False,
            lr_1d=args.lr,
            # betas_1d=(args.beta1, args.beta2),
            weight_decay_1d=args.weight_decay,
        )
    elif optimizer_name == "mars_m":
        muon_params = []
        adamw_params = []
        for group in param_groups:
            if group.get("is_proj_params", False):
                muon_params.extend(group["params"])
            else:
                adamw_params.extend(group["params"])
        optimizer = MARS_M(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            momentum=args.momentum,
            ns_steps=args.muon_ns_steps,
            adamw_params=adamw_params,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=args.eps,
            gamma=args.mars_gamma,
            # clip_c=False,
            # is_approx=True,
        )
    elif "apollo" in optimizer_name:
        for group in param_groups:
            if group.get("is_proj_params", False):
                if "mini" in optimizer_name:
                    group["rank"] = args.rank
                else:
                    group["rank"] = int(args.density * args.n_embd)
                print("\n"*3)
                print("-"*30)
                print(f"APOLLO Rank: {group['rank']}")
                print("-"*30)
                print("\n"*3)
                group["update_proj_gap"] = args.update_gap
                group["scale"] = args.apollo_scale
                group["proj"] = args.apollo_proj
                group["scale_type"] = args.apollo_scale_type

                group["proj_type"] = args.proj_side
        optimizer = APOLLOAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, scale_front=args.apollo_scale_front)
    elif optimizer_name in ("ldadam", "ldadamw"):
        for group in param_groups:
            if group.get("is_proj_params", False):
                group["enable_lowrank"] = True
                group["rank"] = int(args.density * args.n_embd)
                print("\n"*3)
                print("-"*30)
                print(f"LDAdam Rank: {group['rank']}")
                print("-"*30)
                print("\n"*3)
                group["rho"] = args.ldadam_rho
                group["proj_method"] = args.ldadam_proj_method
                group["error_feedback"] = args.ldadam_error_feedback

                group["proj_type"] = args.proj_side
            else:
                group["enable_lowrank"] = False
        optimizer = LDAdamW(param_groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
    elif optimizer_name == "fira_adamw":
        for group in param_groups:
            if group.get("is_proj_params", False):
                group["rank"] = int(args.density * args.n_embd)
                print("\n"*3)
                print("-"*30)
                print(f"Fira Rank: {group['rank']}")
                print("-"*30)
                print("\n"*3)
                group["update_proj_gap"] = args.update_gap
                group["alpha"] = args.fira_alpha

                group["proj_side"] = args.proj_side
                group["proj_type"] = args.proj_type

                group["reset_statistics"] = args.reset_statistics
        optimizer = FiraAdamW(param_groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
    elif optimizer_name == "galore_adafactor":
        for group in param_groups:
            if group.get("is_proj_params", False):
                group["rank"] = int(args.density * args.n_embd)
                print("\n"*3)
                print("-"*30)
                print(f"Galore Adafactor Rank: {group['rank']}")
                print("-"*30)
                print("\n"*3)
                group["update_proj_gap"] = args.update_gap
                group["scale"] = args.proj_params_lr_scale
                group["proj_type"] = args.proj_side
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    elif optimizer_name == "adamem":
        for group in param_groups:
            if group.get("is_proj_params", False):
                group["rank"] = int(args.density * args.n_embd)
                print("\n"*3)
                print("-"*30)
                print(f"AdaMeM Adafactor Rank: {group['rank']}")
                print("-"*30)
                print("\n"*3)
                group["update_proj_gap"] = args.update_gap
                group["scale"] = args.proj_params_lr_scale
                group["proj_type"] = args.proj_side

                group["use_adamem"] = True
                group["adamem_type"] = args.adamem_type
                group["adamem_rms"] = args.adamem_rms
                group["adamem_relative_lr"] = args.adamem_relative_lr
                group["adamem_reduce_op"] = args.adamem_reduce_op
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    elif optimizer_name == "adamem_orig":
        for group in param_groups:
            if group.get("is_proj_params", False):
                group["rank"] = int(args.density * args.n_embd)
                print("\n"*3)
                print("-"*30)
                print(f"AdaMeM Adafactor Rank: {group['rank']}")
                print("-"*30)
                print("\n"*3)
                group["update_proj_gap"] = args.update_gap
                group["scale"] = args.proj_params_lr_scale
                group["proj_type"] = args.proj_side
                group["adamem_reduce_op"] = args.adamem_reduce_op

        optimizer = AdaMeM(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,

            adamem_relative_lr=args.adamem_relative_lr,
            use_momentum_to_update_variance=args.use_momentum_to_update_variance,
        )

    elif optimizer_name == "galore_adamw":
        # redefine way to call galore_adamw
        optimizer = GaloreAdamW(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # galore specific
            proj_side=args.proj_side,
            proj_type=args.proj_type,
            # adam specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif optimizer_name == "coord_adamw":
        optimizer = CoordAdamW(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            coord_choice=args.coord_choice,
            # adam specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif optimizer_name == "block_adamw":
        optimizer = BlockAdamW(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            block_order=args.block_order,
            # adam specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif optimizer_name == "adalayer":
        for group in param_groups:
            group["sqrt_numel"] = args.sqrt_numel
        optimizer = Adalayer(param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif optimizer_name == "block_adalayer":
        for group in param_groups:
            group["sqrt_numel"] = args.sqrt_numel
        optimizer = BlockAdalayer(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            block_order=args.block_order,
            # adalayer specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif optimizer_name == "lion":
        optimizer = Lion(param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_name == "galore_lion":
        optimizer = GaloreLion(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # galore specific
            proj_side=args.proj_side,
            proj_type=args.proj_type,
            # lion specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_name == "coord_lion":
        optimizer = CoordLion(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            coord_choice=args.coord_choice,
            # lion specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_name == "block_lion":
        optimizer = BlockLion(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            block_order=args.block_order,
            # lion specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay)
    # implement sgd
    elif optimizer_name == "sgd":
        optimizer = SGD(param_groups, lr=args.lr, momentum=args.beta1, dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov, sign_update=args.sgd_sign_update)
    elif optimizer_name == "galore_sgd":
        optimizer = GaloreSGD(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # galore specific
            proj_side=args.proj_side,
            proj_type=args.proj_type,
            # sgd specific
            lr=args.lr, momentum=args.beta1, dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov, sign_update=args.sgd_sign_update)
    elif optimizer_name == "coord_sgd":
        optimizer = CoordSGD(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # galore specific
            coord_choice=args.coord_choice,
            # sgd specific
            lr=args.lr, momentum=args.beta1, dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov, sign_update=args.sgd_sign_update)
    elif optimizer_name == "block_sgd":
        optimizer = BlockSGD(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            block_order=args.block_order,
            # lion specific
            lr=args.lr, momentum=args.beta1, dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov, sign_update=args.sgd_sign_update)
    elif optimizer_name == "lora":
        # Resolve base optimizer class
        lora_base_map = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
        }
        base_cls = lora_base_map[args.lora_base_opt]

        for group in param_groups:
            if group.get("is_proj_params", False):
                group["rank"] = args.lora_rank if args.lora_rank > 0 else int(args.density * args.n_embd)
                print("\n" * 3)
                print("-" * 30)
                print(f"LoRA Rank: {group['rank']}")
                print("-" * 30)
                print("\n" * 3)

        # Build base-optimizer kwargs depending on the chosen base
        base_kwargs = dict(lr=args.lr, weight_decay=args.weight_decay)
        if args.lora_base_opt in ("adamw", "adam"):
            base_kwargs.update(betas=(args.beta1, args.beta2), eps=args.eps)
        else:  # sgd
            base_kwargs.update(momentum=args.momentum, dampening=args.dampening, nesterov=args.nesterov)

        optimizer = LoRAOptimizer(
            param_groups,
            optimizer_cls=base_cls,
            rank=args.lora_rank if args.lora_rank > 0 else int(args.density * args.n_embd),
            lora_alpha=args.lora_alpha,
            **base_kwargs,
        )
    elif optimizer_name == "lora_rite":
        for group in param_groups:
            if group.get("is_proj_params", False):
                group["rank"] = args.lora_rank if args.lora_rank > 0 else int(args.density * args.n_embd)
                print("\n" * 3)
                print("-" * 30)
                print(f"LoRA-Rite Rank: {group['rank']}")
                print("-" * 30)
                print("\n" * 3)

        optimizer = LoRARiteOptimizer(
            param_groups,
            rank=args.lora_rank if args.lora_rank > 0 else int(args.density * args.n_embd),
            lora_alpha=args.lora_alpha,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            clip_unmagnified_grad=args.lora_rite_clip_grad,
            update_capping=args.lora_rite_update_capping,
            update_skipping=args.lora_rite_update_skipping,
            apply_escape=args.lora_rite_apply_escape,
            balance_param=args.lora_rite_balance_param,
        )
    elif optimizer_name in ("loro", "loro_adpt"):
        # LORO param_groups are already constructed with "type" keys
        # by get_loro_param_groups() in main.py.  The model reference
        # is needed for _loro_step() which iterates over LowRank modules.
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = LOROAdamW(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            correct_bias=True,
            no_deprecation_warning=True,
            loro_type=args.loro_type,
            model=raw_model,
            use_exact_loro=args.use_exact_loro,
        )
    elif optimizer_name == "swan":
        ns_func_mapping = {
            'jordan': 'jordan',
            'express_orig': 'polar_orig',
            'express_modified': 'polar_mod',
            '5777_left_1e_3': '5777',
            '5779_left_15e_4': '5779',
            'cesista': 'jordan',
            'svd': 'jordan', 
        }
        use_nesterov = args.nesterov and (args.beta1 > 0)
        ns_func = ns_func_mapping.get(args.newton_schulz_func, 'jordan')
        optimizer = SWAN(
            param_groups, 
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.beta1,
            dampening=args.dampening,
            nesterov=use_nesterov,
            sign_update=args.sgd_sign_update,
            k=args.muon_ns_steps,
            beta=args.swan_ns_step_size,
            ns_func=ns_func,
            epsilon=args.eps,
            rescale=not getattr(args, 'swan_no_rescale', False),
            min_numel_whitening=args.swan_min_numel_whitening,
        )
    elif optimizer_name == "badam":
        raw_model = model.module if hasattr(model, "module") else model
        named_params = list(raw_model.named_parameters())

        # Build block_prefix_list: one list of prefixes per block.
        # Each block covers badam_block_size consecutive transformer layers.
        block_size = max(1, args.badam_block_size)
        n_layers = args.n_layer
        block_prefix_list = []
        for block_start in range(0, n_layers, block_size):
            block_prefixes = []
            for idx in range(block_start, min(block_start + block_size, n_layers)):
                block_prefixes.append(f"transformer.h.{idx}.")
            block_prefix_list.append(block_prefixes)

        # Always-active modules (embedding, final norm, lm_head)
        active_modules = ["transformer.wte", "transformer.ln_f", "lm_head"]

        base_optimizer = torch.optim.AdamW(
            param_groups,
            betas=(args.beta1, args.beta2),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.eps,
            foreach=False,
            fused=False,
        )
        optimizer = BlockOptimizer(
            base_optimizer=base_optimizer,
            named_parameters_list=named_params,
            block_prefix_list=block_prefix_list,
            switch_block_every=args.update_gap,
            switch_mode=args.badam_switch_mode,
            active_modules=active_modules,
            verbose=args.badam_verbose,
        )
    elif optimizer_name == "coap_adamw":
        optimizer = COAPAdamW(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            density=args.density,
            update_interval=args.coap_update_interval,
            reproject_factor=args.coap_reproject_factor,
            restore_state=args.coap_restore_state,
            scale=args.coap_scale,
        )
    elif optimizer_name == "cosmos":
        optimizer = COSMOS(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            density=args.density,
            lr_ratio=args.cosmos_lr_ratio,
            gamma=args.cosmos_gamma,
            nestrov=args.cosmos_nestrov,
        )
    elif optimizer_name == "sumo":
        rank = max(1, int(args.density * args.n_embd))
        print("\n" * 3)
        print("-" * 30)
        print(f"SUMO Rank: {rank}")
        print("-" * 30)
        print("\n" * 3)
        for group in param_groups:
            if group.get("is_proj_params", False):
                group["group_name"] = "sumo_params"
                group["rank"] = rank
                group["scale"] = args.sumo_scale
                group["proj_type"] = args.sumo_proj_type
            else:
                group["group_name"] = "adam_params"
        optimizer = SUMO(
            param_groups,
            lr=args.lr,
            beta=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            rank=rank,
            update_proj_gap=args.update_gap,
            alpha=args.sumo_alpha,
            gamma=args.sumo_gamma,
            nesterov=args.nesterov,
            momentum=args.momentum,
            norm_growth_limiter=args.sumo_norm_growth_limiter,
            gradient_perpendicular_scale=args.sumo_gradient_perpendicular_scale,
            lr_adam=args.sumo_lr_adam,
            weight_decay_adam=args.sumo_weight_decay_adam,
            eps=args.eps,
            correct_bias=True,
            no_deprecation_warning=True,
        )
    elif optimizer_name == "slim_adam":
        raw_model = model.module if hasattr(model, "module") else model
        # Use explicit rules JSON if provided, otherwise fall back to the bundled layer map.
        rules_json = args.slim_adam_rules_json if args.slim_adam_rules_json else None
        layer_map  = None if rules_json else (args.slim_adam_layer_map or SLIM_ADAM_DEFAULT_LAYER_MAP)
        optimizer = SlimAdamW(
            named_parameters=raw_model.named_parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            rules_json_path=rules_json,
            layer_map_path=layer_map,
            verbose=args.slim_adam_verbose,
        )
    elif optimizer_name == "adam_mini":
        # Adam-mini takes named_parameters directly and builds its own param groups.
        # This model uses fused c_attn (combined Q/K/V) and c_proj for both attn output
        # and MLP down-proj. Adam-mini will classify:
        #   - transformer.wte          → embd_names  (one lr per token)
        #   - lm_head                  → output_names (one lr per token)
        #   - *.mlp.w12 / *.mlp.c_proj → mlp_names   (one lr per neuron, via "mlp" substring)
        #   - *.attn.c_attn            → other blocks (single lr; fused QKV can't split by head)
        #   - *.attn.c_proj            → other blocks (single lr)
        #   - layer norms (ln_1, ln_2) → other blocks (weight decay=0 by name)
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = Adam_mini(
            named_parameters=raw_model.named_parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            dim=args.n_embd,
            n_heads=args.n_head,
            n_kv_heads=args.adam_mini_n_kv_heads if args.adam_mini_n_kv_heads > 0 else None,
            verbose=args.adam_mini_verbose,
        )
    else:
        raise ValueError(f"Optimizer {args.opt} not supported")
    return optimizer
