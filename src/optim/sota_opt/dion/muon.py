import math
import torch
import torch.distributed as dist
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union, Dict
# import dion
from . import newton_schulz_funcs

from functools import partial

from .newton_schulz_triton import newton_schulz_triton
from .opt_utils import (
    AsyncTask,
    AsyncRuntime,
    to_local,
    create_param_batches,
    pad_batch,
)
from .scalar_opts import (
    lion_update_foreach,
    adamw_update_foreach,
    adan_update_foreach,
)


class Muon(Optimizer):
    """
    Distributed Muon optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For Muon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        mu: Momentum factor for Muon algorithm.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr: How to adjust the learning rate for Muon updates ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate.
        flatten: Whether to flatten 3D+ tensors to 2D for Muon updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is `func(input: Tensor, epsilon: float) -> Tensor`.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    FSDP2 Muon uses all-to-all communications: https://www.essential.ai/blog/infra
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        adan_betas: Tuple[float, float, float] = (0.98, 0.92, 0.99),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_triton: bool = False,
        dampening: float = 0.0,
        newton_schulz_func_name: str = "jordan",
        muon_ns_steps: int = 5,
        pre_orth_update_name: str = "default",
        headwise=False, # requires "is_qkv_params" and "attention_head_size" keys in some param_group
        adamw_lr_scale=0.2, # useful for spectral adjust_lr, since in this muon requires higher lr
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )
        if len(adan_betas) != 3 or any(b < 0.0 for b in adan_betas):
            raise ValueError(f"Invalid adan_betas: {adan_betas}")

        if adjust_lr == "rms_norm":
            assert adamw_lr_scale == 1.0, "You don't usually want to have different learning rates for muon and adamw with `rms_norm` adjust_lr option."

        if newton_schulz_func_name == "cesista":
            newton_schulz_func = None
        elif newton_schulz_func_name == "jordan":
            newton_schulz_func = newton_schulz_funcs.zeropower_via_newtonschulz5_jordan
        elif newton_schulz_func_name == "svd":
            # TODO
            raise NotImplementedError("TODO")
        elif newton_schulz_func_name == "express_orig":
            newton_schulz_func = partial(newton_schulz_funcs.PolarExpressOrig, steps=muon_ns_steps)
        elif newton_schulz_func_name == "express_modified":
            newton_schulz_func = partial(newton_schulz_funcs.PolarExpressModified, steps=muon_ns_steps)
        elif newton_schulz_func_name == "5777_left_1e_3":
            newton_schulz_func = newton_schulz_funcs.zeropower_5777_left_1e_3
        elif newton_schulz_func_name == "5779_left_15e_4":
            newton_schulz_func = newton_schulz_funcs.zeropower_5779_left_15e_4
        else:
            raise ValueError("unknown newton_schulz_func name")
        
        if pre_orth_update_name == "default":
            pre_orth_update_func = muon_update_pre_orthogonalize
        elif pre_orth_update_name == "ns_adan":
            pre_orth_update_func = muon_adan_update_pre
        elif pre_orth_update_name == "ema":
            pre_orth_update_func = muon_ema_update_pre_orthogonalize
        else:
            raise ValueError(f"Unknown pre_orth_update_name: {pre_orth_update_name}")

        # Default arguments for each param group
        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            algorithm="muon",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
            headwise=headwise,
            dampening=dampening,
            adan_beta1=adan_betas[0],
            adan_beta2=adan_betas[1],
            adan_beta3=adan_betas[2],
            pre_fn_name=pre_orth_update_name,
        )
        super().__init__(params, defaults)
        if headwise:
            assert any("is_qkv_params" in group and "attention_head_size" in group for group in self.param_groups), "`headwise` requires 'is_qkv_params' and 'attention_head_size' keys in some param_group"
            assert not flatten, "`flatten` conflicts with `headwise`"

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh is supported, but got {distributed_mesh.ndim}D. For HSDP, provide the 1D sharded sub-mesh."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. Expected DeviceMesh or ProcessGroup."
            )
        self._distributed_mesh = distributed_mesh
        self._pre_orth_update_name = pre_orth_update_name

        # Newton-Schulz configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(
                    f"newton_schulz_func must be a callable function, got {type(newton_schulz_func)}"
                )
            self._newton_schulz_func = newton_schulz_func
        elif use_triton:
            self._newton_schulz_func = newton_schulz_triton
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5
        
        # Pre-Orth configuration
        if pre_orth_update_func is not None:
            if not callable(pre_orth_update_func):
                raise TypeError(
                    f"pre_orth_update_func must be a callable function, got {type(pre_orth_update_func)}"
                )
            self._pre_orth_update_func = pre_orth_update_func
        else:
            self._pre_orth_update_func = muon_update_pre_orthogonalize
        
        for group in self.param_groups:
            if group["algorithm"] in ["adamw", "lion"]:
                group["lr"] *= adamw_lr_scale

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_groups = []
        lion_groups = []
        adamw_groups = []
        adan_groups = []

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "muon":
                muon_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            elif algo == "adan":
                adan_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        muon_tasks = self._create_muon_tasks(muon_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)
        adan_tasks = self._create_adan_tasks(adan_groups)

        all_tasks = chain(muon_tasks, lion_tasks, adamw_tasks, adan_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=4)
        runtime.run()

        return loss

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        if not state:
            if algo == "adan":
                # Adan backup
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_diff"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
                state["grad_prev"] = torch.zeros_like(param)
            elif algo == "adamw":
                state["variance"] = torch.zeros_like(param)
                state["momentum"] = torch.zeros_like(param)

            if self._pre_orth_update_name == "ns_adan":
                # NSAdan
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_diff"] = torch.zeros_like(param)
                state["grad_prev"] = torch.zeros_like(param)
            else:
                state["momentum"] = torch.zeros_like(param)
                # if algo == "adamw":
                #     state["variance"] = torch.zeros_like(param)
        return state

    def _create_dummy_state(self, param: Tensor, algo_name: str) -> dict:
        dummy_state = {}
        if algo_name == "muon":
            if self._pre_orth_update_name == "ns_adan":
                dummy_state["exp_avg"] = torch.zeros_like(param)
                dummy_state["exp_avg_diff"] = torch.zeros_like(param) 
                dummy_state["grad_prev"] = torch.zeros_like(param)
            else:
                dummy_state["momentum"] = torch.zeros_like(param)
        elif algo_name == "adamw":
            dummy_state["momentum"] = torch.zeros_like(param)
            dummy_state["variance"] = torch.zeros_like(param)
        elif algo_name == "adan":
            dummy_state["exp_avg"] = torch.zeros_like(param)
            dummy_state["exp_avg_diff"] = torch.zeros_like(param)
            dummy_state["exp_avg_sq"] = torch.zeros_like(param)
            dummy_state["grad_prev"] = torch.zeros_like(param)
        return dummy_state

    def _create_muon_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "muon",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of Muon matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Muon optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Wrap hyperparameters in tensors for torch.compile
            # lr = torch.tensor(group["lr"])
            # mu = torch.tensor(group["mu"])
            # weight_decay = torch.tensor(group["weight_decay"])
            # epsilon = torch.tensor(group["epsilon"])
            # nesterov = group["nesterov"]
            # flatten = group["flatten"]
            # adjust_lr = group["adjust_lr"]
            # attention_head_size = group.get("attention_head_size", None)
            # headwise = group["headwise"] and bool(attention_head_size)
            # dampening = group["dampening"]

            hyperparams = {
                'lr': torch.tensor(group["lr"]),
                'mu': torch.tensor(group["mu"]),
                'weight_decay': torch.tensor(group["weight_decay"]),
                'epsilon': torch.tensor(group["epsilon"]),
                'beta1': torch.tensor(group["beta1"]),
                'beta2': torch.tensor(group["beta2"]),
                'nesterov': group["nesterov"],
                'flatten': group["flatten"],
                'adjust_lr': group["adjust_lr"],
                'headwise': group["headwise"] and bool(group.get("attention_head_size")),
                'attention_head_size': group.get("attention_head_size", None),
                'dampening': torch.tensor(group["dampening"]),
            }

            # Create batches of parameters of size self._world_size
            for params in create_param_batches(
                group_params, batch_size=self._world_size
            ):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, algo_name) for p in params]
                # momentums = [s["momentum"] for s in states]

                padded_X = pad_batch(params, self._world_size)
                padded_G = pad_batch(gradients, self._world_size)

                states_padded = states.copy()
                while len(states_padded) < self._world_size:
                    dummy_param = padded_X[len(states_padded)]
                    dummy_state = self._create_dummy_state(dummy_param, algo_name)
                    states_padded.append(dummy_state)

                # Get sharding dimension
                sharded_mesh_dim = None
                sharded_tensor_dim = None
                if isinstance(params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "Must create optimizer with DeviceMesh if using DTensor parameters."
                        )

                    # Find the sharded placement and get its mesh and tensor dimensions
                    # Skip any Shard() placements on size-1 mesh dimension = Replicate()
                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and params[0].device_mesh.size(i) > 1
                    ]
                    if len(shard_placements) == 1:
                        sharded_mesh_dim = shard_placements[0][0]
                        sharded_tensor_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError(
                            "Muon does not support parameters with multiple sharded dimensions."
                        )

                    # Check that the sharded mesh dimension matches optimizer's device mesh
                    if (
                        sharded_mesh_dim is not None
                        and params[0].device_mesh.get_group(sharded_mesh_dim)
                        != self._process_group
                    ):
                        raise RuntimeError(
                            f"Got DTensor sharded over mesh dimension {sharded_mesh_dim} different from the optimizer's device mesh"
                        )

                yield AsyncTask(
                    muon_update_batch_async(
                        # X=pad_batch(params, self._world_size),
                        # G=pad_batch(gradients, self._world_size),
                        X = padded_X,
                        G = padded_G,
                        # M=pad_batch(momentums, self._world_size),
                        # states=states,
                        states=states_padded,
                        # lr=lr,
                        # momentum=mu,
                        # weight_decay=weight_decay,
                        # epsilon=epsilon,
                        # nesterov=nesterov,
                        # flatten=flatten,
                        # adjust_lr=adjust_lr,
                        hyperparams=hyperparams,
                        device_rank=self._device_rank,
                        world_size=self._world_size,
                        shard_dim=sharded_tensor_dim,
                        process_group=self._process_group,
                        newton_schulz_func=self._newton_schulz_func,
                        pre_orth_update_func=self._pre_orth_update_func,
                        # headwise=headwise,
                        # attention_head_size=attention_head_size,
                        # dampening=dampening
                    )
                )

    def _create_lion_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "lion",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for Lion updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                )
            )

    def _create_adan_tasks(
            self,
            param_groups: List[dict],
            algo_name: str = "adan",
        ) -> Generator["AsyncTask", None, None]:
            """
            Helper function to generate AsyncTask objects for Adan updates.
            """
            for group in param_groups:
                assert group["algorithm"] == algo_name

                # Get parameters and optimizer states
                params = [p for p in group["params"] if p.grad is not None]
                if not params:
                    continue

                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, algo_name) for p in params]

                exp_avgs = [s["exp_avg"] for s in states]
                exp_avg_diffs = [s["exp_avg_diff"] for s in states]
                exp_avg_sqs = [s["exp_avg_sq"] for s in states]
                grad_prevs = [s["grad_prev"] for s in states]

                # Wrap hyperparameters in tensors for torch.compile
                lr = torch.tensor(group["lr"])
                beta1 = torch.tensor(group["adan_beta1"])
                beta2 = torch.tensor(group["adan_beta2"])
                beta3 = torch.tensor(group["adan_beta3"])
                weight_decay = torch.tensor(group["weight_decay"])
                epsilon = torch.tensor(group["epsilon"])
                step = torch.tensor(group["step"])

                yield AsyncTask(
                    adan_update_foreach_async(
                        X=to_local(params),
                        G=to_local(gradients),
                        M=to_local(exp_avgs),
                        V=to_local(exp_avg_diffs),
                        N=to_local(exp_avg_sqs),
                        D=to_local(grad_prevs),
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        beta3=beta3,
                        weight_decay=weight_decay,
                        step=step,
                        epsilon=epsilon.item(),
                    )
                )

def muon_update_batch_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    # M: List[Tensor],  # Momentum buffer (modified in place)
    states: Optional[List[dict]],
    # lr: Tensor,  # Learning rate (scalar tensor)
    # momentum: Tensor,  # Momentum factor (scalar tensor)
    # weight_decay: Tensor,  # Weight decay (scalar tensor)
    # epsilon: Tensor,  # Epsilon (scalar tensor)
    # nesterov: bool,  # Whether to use Nesterov momentum
    # dampening: float, 
    # flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    hyperparams: Dict,  # All hyperparameters
    # adjust_lr: Optional[str],  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    shard_dim: Optional[int] = None,  # Shard dimension for DTensor (if applicable)
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    pre_orth_update_func: Optional[Callable] = None
    # headwise=False,
    # attention_head_size=None
) -> Generator[None, None, None]:
    """
    Batched version of Muon update. Batch size should be equal to number of GPUs.
    All tensors in a batch should have identical shape, sharding, and dtype.
    Identical hyperparameters are used for all tensors in the batch.
    """

    assert len(X) == len(G)
    assert len(X) == len(states)
    assert len(X) == world_size

    lr = hyperparams["lr"]
    # momentum = hyperparams["momentum"]
    weight_decay = hyperparams["weight_decay"]
    epsilon = hyperparams["epsilon"]
    headwise = hyperparams["headwise"]
    attention_head_size = hyperparams["attention_head_size"]
    adjust_lr = hyperparams["adjust_lr"]
    flatten = hyperparams["flatten"]

    # Update momentum and compute the inputs for orthogonalization
    if pre_orth_update_func is None:
        pre_orth_update_func = muon_update_pre_orthogonalize

    U = pre_orth_update_func(
        G=to_local(G),
        states=states,
        hyperparams=hyperparams,
    )

    # Get one whole matrix for each device to orthogonalize
    if shard_dim is not None:
        # Use all-to-all to transform from a batch of shards to a single whole matrix
        # https://www.essential.ai/blog/infra
        assert (
            process_group is not None
        ), "process_group must be provided for sharded DTensors"
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert not isinstance(U[0], DTensor), "U should contain local shards"
        assert (
            X[0].size(shard_dim) % world_size == 0
        ), f"Shard dimension {shard_dim} size {X[0].size(shard_dim)} is not divisible by world size {world_size}."

        # Allocate buffers to receive shards of one whole matrix from other devices
        single_matrix_shards = [torch.empty_like(u) for u in U]

        # Redistribute the shards to form one unique full tensor on each device
        work = dist.all_to_all(
            single_matrix_shards, U, group=process_group, async_op=True
        )
        yield
        work.wait()

        # Concatentate shards to form a whole matrix to orthogonalize
        single_matrix = torch.cat(single_matrix_shards, dim=shard_dim)
        if headwise:
            num_attention_heads = single_matrix.shape[0] // attention_head_size
            single_matrix = single_matrix.view(num_attention_heads, single_matrix.shape[0] // num_attention_heads, single_matrix.shape[1])
        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )
        if headwise:
            single_matrix = single_matrix.view(-1, single_matrix.shape[-1])

        # Split result back into shards
        # Contiguous is needed for all-to-all to work correctly
        single_matrix_shards = [
            x.contiguous()
            for x in torch.tensor_split(single_matrix, world_size, dim=shard_dim)
        ]

        # Redistribute the orthogonalized tensor back to original layout
        work = dist.all_to_all(
            U, single_matrix_shards, group=process_group, async_op=True
        )
        yield
        work.wait()

    else:
        # Matrices are not sharded, so we can directly orthogonalize
        # Get a single matrix corresponding to this device
        single_matrix = U[device_rank]
        assert not isinstance(single_matrix, DTensor)
        if headwise:
            num_attention_heads = single_matrix.shape[0] // attention_head_size
            single_matrix = single_matrix.view(num_attention_heads, single_matrix.shape[0] // num_attention_heads, single_matrix.shape[1])
        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )
        if headwise:
            single_matrix = single_matrix.view(-1, single_matrix.shape[-1])

        if process_group is not None and process_group.size() > 1:
            # Allocate empty tensors to receive updates from other devices
            U = [torch.empty_like(u) for u in U]

            # All gather orthogonalized results from other devices into buffer
            work = dist.all_gather(
                U, single_matrix.contiguous(), group=process_group, async_op=True
            )
            yield
            work.wait()

        else:
            # Single GPU case, no need to gather
            assert world_size == 1
            U = [single_matrix]

    # Compute scaled learning rate
    # Do this before to_local(X) because we use the full tensor shape, not the shard shape
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    # Update model parameters with orthogonalized output
    muon_update_post_orthogonalize(
        X=to_local(X),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )


def adamw_update_foreach_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
) -> Generator[None, None, None]:
    """
    Async wrapper around foreach AdamW update.
    """
    adamw_update_foreach(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon)
    yield


def lion_update_foreach_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
) -> Generator[None, None, None]:
    """
    Async wrapper around foreach Lion update.
    """
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay)
    yield


def adan_update_foreach_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: Tensor,  # First moment buffer (m_t) (modified in place)
    V: Tensor,  # Corrected first moment buffer (v_t) (modified in place)
    N: Tensor,  # Second moment buffer (n_t) (modified in place)
    D: Tensor,  # Previous gradient
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    beta3: Tensor,  # Beta 3 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
) -> Generator[None, None, None]:
    """
    Async wrapper around foreach Adan update.
    """
    adan_update_foreach(X, G, M, V, N, D, lr, beta1, beta2, beta3, weight_decay, step, epsilon)
    yield

# @torch.compile(fullgraph=True)
def muon_adan_update_pre(
    G: List[Tensor],
    states: List[dict],
    hyperparams: Dict,
    # beta1: float,
    # beta2: float,
) -> List[Tensor]:
    """
    Adan-like pre NS update
    """
    beta1 = float(hyperparams['beta1'])
    beta2 = float(hyperparams['beta2'])

    dtype = states[0]["exp_avg"].dtype
    G = [g.to(dtype=dtype) for g in G]

    exp_avgs = [s["exp_avg"] for s in states]
    exp_avg_diffs = [s["exp_avg_diff"] for s in states]
    grad_prevs = [s["grad_prev"] for s in states]

    grad_diffs = torch._foreach_sub(G, grad_prevs)

    # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    torch._foreach_lerp_(exp_avgs, G, weight=1 - beta1)

    # v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t - g_{t-1})
    torch._foreach_lerp_(exp_avg_diffs, grad_diffs, weight=1 - beta2)

    for i in range(len(states)):
        states[i]["grad_prev"].copy_(G[i])

    # m_t + (1 - beta2) * v_t
    scaled_diffs = torch._foreach_mul(exp_avg_diffs, 1 - beta2)
    combined = torch._foreach_add(exp_avgs, scaled_diffs)

    return [c.to(dtype=torch.bfloat16) for c in combined]

# @torch.compile(fullgraph=True)
def muon_update_pre_orthogonalize(
    G: List[Tensor],
    states: List[dict],
    hyperparams: Dict,
    # M: List[Tensor],
    # momentum: Tensor,
    # nesterov: bool,
    # dampening: float,
) -> List[Tensor]:
    """
    Update momentum with gradient and compute the input to orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """

    momentum = hyperparams['mu']
    nesterov = hyperparams['nesterov']
    dampening = float(hyperparams['dampening'])

    M = [s["momentum"] for s in states]
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Update momentum with new gradient
    torch._foreach_mul_(M, momentum)
    if dampening != 0:
        damped_G = torch._foreach_mul(G, 1.0 - dampening)
        torch._foreach_add_(M, damped_G)
    else:
        torch._foreach_add_(M, G)

    if nesterov:
        U = torch._foreach_mul(M, momentum)
        torch._foreach_add_(U, G)
    else:
        U = M

    # Convert to bfloat16 before communication
    U = [u.to(dtype=torch.bfloat16) for u in U]

    return U


@torch.compile(fullgraph=True)
def muon_ema_update_pre_orthogonalize(
    G: List[Tensor],
    states: List[dict],
    hyperparams: Dict,
    # M: List[Tensor],
    # momentum: Tensor,
    # nesterov: bool,
    # dampening: float,
) -> List[Tensor]:
    """
    Update momentum with gradient and compute the input to orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """

    mu = hyperparams['mu']
    nesterov = hyperparams['nesterov']
    dampening = float(hyperparams['dampening'])

    M = [s["momentum"] for s in states]
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Update momentum with new gradient
    torch._foreach_mul_(M, mu)
    G_scale = torch._foreach_mul(G, 1-mu)
    torch._foreach_add_(M, G_scale)

    if nesterov:
        U = torch._foreach_mul(M, mu)
        torch._foreach_add_(U, G_scale)
    else:
        U = M

    # Convert to bfloat16 before communication
    U = [u.to(dtype=torch.bfloat16) for u in U]

    return U


@torch.compile(fullgraph=True)
def muon_update_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
):
    """
    Apply weight decay and weight update after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    # Apply weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Weight update
    U = torch._foreach_mul(U, adjusted_lr)
    torch._foreach_sub_(X, U)


def muon_update_newton_schulz(
    X: Tensor,
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
) -> Tensor:
    """
    Flatten the input tensor if needed and call the Newton-Schulz function.
    """
    original_shape = X.shape
    if flatten and X.ndim >= 3:
        # Flatten 3D+ tensors to 2D matrix
        X = X.flatten(start_dim=1)
    elif X.ndim >= 4:
        # Given 4D+ batch, flatten to 3D batch
        X = X.flatten(end_dim=-3)

    return newton_schulz_func(X, epsilon=epsilon).reshape(original_shape)


def adjust_lr_rms_norm(lr, param_shape):
    # Adjust learning rate for constant element-wise RMS norm
    # https://arxiv.org/abs/2502.16982
    A, B = param_shape[:2]
    adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


def adjust_lr_spectral_norm(lr, param_shape):
    # Adjust from spectral norm 1 to RMS operator norm 1
    # https://arxiv.org/abs/2310.17813
    fan_out, fan_in = param_shape[:2]
    adjusted_lr = lr * math.sqrt(fan_out / fan_in)
    return adjusted_lr


@torch.compile(fullgraph=True)
def zeropower_via_newtonschulz5(G: Tensor, epsilon: float = 1e-7):
    """
    Newton-Schulz iteration to approximate the orthogonalization of X.
    """
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
