"""Dynamic installation that leaves the baseline implementation untouched."""

from __future__ import annotations

from functools import partial

from .ops import VALID_CACHE_POLICIES, custom_tucker_linear


def install_custom_backward(*, cache_policy: str = "persistent") -> None:
    if cache_policy not in VALID_CACHE_POLICIES:
        raise ValueError(
            f"cache_policy must be one of {VALID_CACHE_POLICIES}, got {cache_policy!r}"
        )
    import models.tucker_chunked as target

    target.chunked_tucker_linear = partial(
        custom_tucker_linear, cache_policy=cache_policy
    )
