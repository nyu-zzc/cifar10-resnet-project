"""Strict adapter for the supplied, unmodified Efficient ResNets architecture.

Only the exact vendor_project1_model.py source copy is imported. No supplied checkpoint is read.
The returned object is the author's model itself, so state keys and forward
operations gain no wrapper prefix or replacement initialization.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from torch import nn

PARAMETER_LIMIT = 5_000_000
ARCHITECTURE = "efficient_resnets_project1"
AUTHOR_SOURCE = Path(__file__).resolve().with_name("vendor_project1_model.py")


def _author_module():
    name = f"{__package__}._author_project1_model"
    if name not in sys.modules:
        specification = importlib.util.spec_from_file_location(name, AUTHOR_SOURCE)
        if specification is None or specification.loader is None:
            raise ImportError(f"Cannot load author source: {AUTHOR_SOURCE}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
    return sys.modules[name]


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_model(config=None) -> nn.Module:
    options = dict(config or {})
    unknown = options.keys() - {"architecture", "num_classes"}
    if unknown:
        raise ValueError(f"Unknown author-model options: {sorted(unknown)}")
    architecture = options.get("architecture", ARCHITECTURE)
    num_classes = options.get("num_classes", 10)
    if architecture != ARCHITECTURE or type(num_classes) is not int or num_classes != 10:
        raise ValueError("Only efficient_resnets_project1 with exactly 10 classes is supported")
    model, author_count = _author_module().project1_model()
    count = count_parameters(model)
    if count != int(author_count) or count != 4_697_742 or count > PARAMETER_LIMIT:
        raise ValueError(f"Author architecture parameter count changed: {count}")
    model.config = {"architecture": ARCHITECTURE, "num_classes": 10}
    return model
