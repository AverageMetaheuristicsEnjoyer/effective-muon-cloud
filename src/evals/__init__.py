from data.utils import get_tokenizer

from .downstream import DownstreamEvaluator
from .lm import AuxiliaryLMEvaluator


def build_evaluators(cfg, tokenizer=None):
    downstream_evaluator = None
    lm_evaluator = None

    needs_tokenizer = (
        DownstreamEvaluator.is_enabled(cfg) or AuxiliaryLMEvaluator.is_enabled(cfg)
    )
    if needs_tokenizer and tokenizer is None:
        tokenizer = get_tokenizer(cfg)

    tokenizer_identifier = None
    if tokenizer is not None:
        tokenizer_identifier = getattr(tokenizer, "name_or_path", getattr(cfg, "tokenizer", "gpt2"))

    if DownstreamEvaluator.is_enabled(cfg):
        downstream_evaluator = DownstreamEvaluator(
            cfg=cfg,
            tokenizer=tokenizer,
            tokenizer_identifier=tokenizer_identifier,
        )

    if AuxiliaryLMEvaluator.is_enabled(cfg):
        lm_evaluator = AuxiliaryLMEvaluator(
            cfg=cfg,
            tokenizer=tokenizer,
        )

    return downstream_evaluator, lm_evaluator
