"""DEPRECATED: Use fabricpc.training.train directly.

All multi-GPU functionality has been consolidated into train.py.
The unified train_pcn() auto-detects devices and uses pmap when
multiple devices are available. This module will be removed in a
future release.
"""

import warnings

from fabricpc.training.train import (
    evaluate_pcn as _evaluate_pcn,
)
from fabricpc.training.train import (
    evaluate_transformer as _evaluate_transformer,
)
from fabricpc.training.train import (
    train_pcn as _train_pcn,
)


def _deprecated(name):
    warnings.warn(
        f"fabricpc.training.multi_gpu.{name} is deprecated. Import from fabricpc.training instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def train_pcn_multi_gpu(*args, **kwargs):
    """Deprecated: Use train_pcn() instead."""
    _deprecated("train_pcn_multi_gpu")
    params, _, _ = _train_pcn(*args, **kwargs)
    return params


def evaluate_pcn_multi_gpu(*args, **kwargs):
    """Deprecated: Use evaluate_pcn() instead."""
    _deprecated("evaluate_pcn_multi_gpu")
    return _evaluate_pcn(*args, **kwargs)


def evaluate_transformer_multi_gpu(*args, **kwargs):
    """Deprecated: Use evaluate_transformer() instead."""
    _deprecated("evaluate_transformer_multi_gpu")
    return _evaluate_transformer(*args, **kwargs)
