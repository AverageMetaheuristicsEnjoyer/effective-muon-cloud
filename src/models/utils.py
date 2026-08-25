from .llama import Llama, RMSNorm
from .base import GPTBase, LayerNorm
from .tucker_linear import replace_all_linears_with_tucker
import torch

BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    LayerNorm,
    RMSNorm,
    torch.nn.Embedding,
)


def get_model(args):
    """Return the right model."""
    tucker_enabled = (
        getattr(args, "linear_parameterization", "dense") == "tucker"
    )
    if tucker_enabled and args.model != "llama":
        raise ValueError(
            "--linear-parameterization tucker currently requires --model llama "
            "because GPTBase ties its lm_head to the token embedding."
        )
    if tucker_enabled and (
        getattr(args, "fp8", False) or getattr(args, "fp8_optim", False)
    ):
        raise ValueError(
            "--linear-parameterization tucker is not compatible with --fp8 "
            "or --fp8-optim yet."
        )
    if getattr(args, "tucker_retract_every_step", False) and not tucker_enabled:
        raise ValueError(
            "--tucker-retract-every-step requires "
            "--linear-parameterization tucker."
        )
    if getattr(args, "tucker_vector_transport", False) and not getattr(
        args, "tucker_retract_every_step", False
    ):
        raise ValueError(
            "--tucker-vector-transport requires --tucker-retract-every-step."
        )
    if getattr(args, "tucker_riemannian_muon", False):
        if not getattr(args, "tucker_retract_every_step", False):
            raise ValueError(
                "--tucker-riemannian-muon requires "
                "--tucker-retract-every-step."
            )
        if not getattr(args, "tucker_vector_transport", False):
            raise ValueError(
                "--tucker-riemannian-muon requires --tucker-vector-transport "
                "so Muon momentum remains in the current tangent space."
            )
    if (
        tucker_enabled
        and getattr(args, "attention_type", "standard") == "tensorized"
        and getattr(args, "tensorized_mode", "reconstruction") == "split_concat"
        and args.sequence_length > 128
    ):
        raise ValueError(
            "All-Linear Tucker with tensorized split_concat is unsafe above "
            "sequence length 128 because o_proj has sequence_length**2 inputs. "
            "Use --tensorized-mode reconstruction for long contexts."
        )

    if args.model == "base":
        model = GPTBase(args)
        if getattr(args, "use_pretrained", "none") != "none":
            model.from_pretrained(args.use_pretrained)
    elif args.model == "llama":
        model = Llama(args)
    else:
        raise KeyError(f"Unknown model '{args.model}'.")

    if tucker_enabled:
        replace_all_linears_with_tucker(model, args)
    return model
