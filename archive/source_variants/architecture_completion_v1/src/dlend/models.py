"""CIFAR-sized preactivation Wide ResNets with a strict 5M parameter budget.

Original implementation inspired by the following methods (no pretrained weights):
Wide Residual Networks: https://arxiv.org/abs/1605.07146
Stochastic Depth: https://arxiv.org/abs/1603.09382
Squeeze-and-Excitation: https://arxiv.org/abs/1709.01507

Fractional width multipliers are intentional. A standard WRN-28-5 has 9,132,634
parameters and is ineligible; WRN-28-3.5 has 4,480,474 before optional SE.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping

from torch import Tensor, cat, nn

PARAMETER_LIMIT = 5_000_000


def count_parameters(model: nn.Module) -> int:
    """Count all distinct parameter tensors, including any frozen parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def enforce_parameter_budget(model: nn.Module) -> int:
    """Do not let freezing parameters disguise an oversized inference model."""
    count = count_parameters(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count > PARAMETER_LIMIT or trainable > PARAMETER_LIMIT:
        raise ValueError(
            f"Model exceeds {PARAMETER_LIMIT:,} parameters: "
            f"{count:,} total, {trainable:,} trainable. "
            "Try depth=28,widen_factor=3.5 or depth=16,widen_factor=5."
        )
    return count


class DropPath(nn.Module):
    """Per-example residual-branch dropout with expectation-preserving scaling.

    This implementation computes the residual branch before masking. It provides
    stochastic-depth regularization, not skipped-computation training speedups.
    """

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("DropPath probability must lie in [0, 1].")
        self.probability = float(probability)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return x
        if self.probability == 1.0:
            return x * 0.0
        keep = 1.0 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1))
        mask.bernoulli_(keep)
        return x * mask / keep


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, ratio: float = 0.25) -> None:
        super().__init__()
        if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
            raise ValueError("SE ratio must lie in (0, 1].")
        hidden = max(1, round(channels * ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1)
        self.expand = nn.Conv2d(hidden, channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.expand(self.activation(self.reduce(self.pool(x)))).sigmoid()
        return x * gate


class ZeroPadShortcut(nn.Module):
    """Parameter-free CIFAR shortcut: strided raw input, symmetrically zero padded.

    Spatial subsampling uses ``x[..., ::stride, ::stride]`` and therefore has
    the same ceil-divided shape as an odd-kernel convolution with same padding.
    Channel narrowing is rejected rather than silently discarding information.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        if out_channels < in_channels:
            raise ValueError("zero_pad shortcuts cannot narrow channels")
        if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
            raise ValueError("Shortcut stride must be a positive integer")
        self.stride = stride
        self.before = (out_channels - in_channels) // 2
        self.after = out_channels - in_channels - self.before

    def forward(self, x: Tensor) -> Tensor:
        sampled = x[:, :, ::self.stride, ::self.stride]
        if self.before == 0 and self.after == 0:
            return sampled
        n, _, h, w = sampled.shape
        return cat((sampled.new_zeros(n, self.before, h, w), sampled,
                    sampled.new_zeros(n, self.after, h, w)), dim=1)


class AverageMaxPool(nn.Module):
    """Concatenate global average and maximum descriptors along channels."""

    def __init__(self) -> None:
        super().__init__()
        self.average = nn.AdaptiveAvgPool2d(1)
        self.maximum = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        return cat((self.average(x), self.maximum(x)), dim=1)


class PreActBlock(nn.Module):
    """BN-ReLU-Conv-BN-ReLU-(Dropout)-Conv-(SE) plus a residual shortcut.

    An identity shortcut carries the original input. A projection shortcut uses
    the preactivated input, matching the usual CIFAR Wide ResNet construction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        se_ratio: float = 0.0,
        kernel_size: int = 3,
        shortcut_type: str = "projection",
        shortcut_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must lie in [0, 1).")
        if not isinstance(kernel_size, int) or isinstance(kernel_size, bool) or kernel_size not in (1, 3, 5, 7):
            raise ValueError("kernel_size must be one of 1, 3, 5, 7")
        if shortcut_type not in ("projection", "avgpool_projection", "zero_pad"):
            raise ValueError("Unknown shortcut_type")
        if type(shortcut_kernel_size) is not int or shortcut_kernel_size not in (1, 3):
            raise ValueError("shortcut_kernel_size must be 1 or 3")
        if shortcut_kernel_size != 1 and shortcut_type != "projection":
            raise ValueError("Nondefault shortcut_kernel_size requires projection shortcuts")
        if shortcut_type == "zero_pad" and out_channels < in_channels:
            raise ValueError("zero_pad shortcuts cannot narrow channels")
        padding = kernel_size // 2
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.se = SqueezeExcitation(out_channels, se_ratio) if se_ratio else nn.Identity()
        self.drop_path = DropPath(drop_path)
        # Preserve the exact original default module structure and RNG order.
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        elif shortcut_type == "projection":
            if shortcut_kernel_size == 1:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        elif shortcut_type == "avgpool_projection":
            pool = nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False) if stride > 1 else nn.Identity()
            self.shortcut = nn.Sequential(pool, nn.Conv2d(in_channels, out_channels, 1, 1, bias=False))
        else:
            self.shortcut = ZeroPadShortcut(in_channels, out_channels, stride)

    def forward(self, x: Tensor) -> Tensor:
        activated = self.relu1(self.bn1(x))
        shortcut_input = x if isinstance(self.shortcut, (nn.Identity, ZeroPadShortcut)) else activated
        shortcut = shortcut_input if isinstance(self.shortcut, nn.Identity) else self.shortcut(shortcut_input)
        out = self.conv1(activated)
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        return shortcut + self.drop_path(self.se(out))


class WideResNet(nn.Module):
    """Three residual stages with widths round([16,32,64] * widen_factor).

    ``depth`` uses the established WRN depth convention, depth = 6*n + 4.
    A 3x3 stride-one stem preserves small-image detail; downsampling occurs only
    in the first block of stages two and three. There is no ImageNet-style pool.
    Explicit ``stage_blocks`` overrides the allocation derived from ``depth``;
    ``effective_depth`` records 2*sum(stage_blocks)+4 for accurate reporting.
    Explicit stage widths similarly override ``widen_factor``. The original
    configuration fields remain unchanged so historical checkpoints resume.
    """

    def __init__(
        self,
        depth: int = 28,
        widen_factor: float = 3.5,
        dropout: float = 0.0,
        drop_path: float = 0.05,
        se_ratio: float = 0.0,
        num_classes: int = 10,
        stage_blocks: list[int] | tuple[int, int, int] | None = None,
        stage_widths: list[int] | tuple[int, int, int] | None = None,
        kernel_sizes: list[int] | tuple[int, int, int] | None = None,
        shortcut_type: str = "projection",
        pool_type: str = "avg",
        shortcut_kernel_size: int = 1,
        pool_size: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(depth, int) or depth < 10 or (depth - 4) % 6:
            raise ValueError("WRN depth must be an integer 6*n+4 with n >= 1.")
        if not math.isfinite(widen_factor) or widen_factor <= 0:
            raise ValueError("widen_factor must be finite and positive.")
        if not 0.0 <= dropout < 1.0 or not 0.0 <= drop_path < 1.0:
            raise ValueError("dropout and drop_path must lie in [0, 1).")
        if not 0.0 <= se_ratio <= 1.0:
            raise ValueError("se_ratio must be zero (disabled) or lie in (0, 1].")
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2.")
        if shortcut_type not in ("projection", "avgpool_projection", "zero_pad"):
            raise ValueError("Unknown shortcut_type")
        if pool_type not in ("avg", "max", "avgmax"):
            raise ValueError("pool_type must be avg, max, or avgmax")
        if type(shortcut_kernel_size) is not int or shortcut_kernel_size not in (1, 3):
            raise ValueError("shortcut_kernel_size must be 1 or 3")
        if shortcut_kernel_size != 1 and shortcut_type != "projection":
            raise ValueError("Nondefault shortcut_kernel_size requires projection shortcuts")
        if type(pool_size) is not int or pool_size not in (0, 4):
            raise ValueError("pool_size must be 0 (original global pool) or 4 (CIFAR32 spatial pool)")
        if pool_size and pool_type != "avg":
            raise ValueError("Nondefault pool_size requires pool_type=avg")

        def triple(values, name):
            if not isinstance(values, (list, tuple)) or len(values) != 3 or any(
                not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in values
            ):
                raise ValueError(f"{name} must contain exactly three positive integers")
            return tuple(values)

        effective_widths = triple(stage_widths, "stage_widths") if stage_widths is not None else tuple(round(base * widen_factor) for base in (16, 32, 64))
        if min(effective_widths) < 1:
            raise ValueError("widen_factor produces an empty residual stage.")
        blocks_per_stage = (depth - 4) // 6
        effective_blocks = triple(stage_blocks, "stage_blocks") if stage_blocks is not None else (blocks_per_stage,) * 3
        effective_kernels = triple(kernel_sizes, "kernel_sizes") if kernel_sizes is not None else (3, 3, 3)
        if any(kernel not in (1, 3, 5, 7) for kernel in effective_kernels):
            raise ValueError("kernel_sizes entries must be one of 1, 3, 5, 7")
        if shortcut_type == "zero_pad" and any(a > b for a, b in zip((16,) + effective_widths, effective_widths)):
            raise ValueError("zero_pad shortcuts cannot narrow channels, including the 16-channel stem")
        self.config = {
            "depth": depth,
            "widen_factor": float(widen_factor),
            "dropout": float(dropout),
            "drop_path": float(drop_path),
            "se_ratio": float(se_ratio),
            "num_classes": num_classes,
        }
        # A legacy config must round-trip without gaining new keys: in-progress
        # checkpoints compare resolved config dictionaries exactly on resume.
        for name, supplied, effective in (("stage_blocks", stage_blocks, effective_blocks),
                                          ("stage_widths", stage_widths, effective_widths),
                                          ("kernel_sizes", kernel_sizes, effective_kernels)):
            if supplied is not None:
                self.config[name] = list(effective)
        if shortcut_type != "projection":
            self.config["shortcut_type"] = shortcut_type
        if pool_type != "avg":
            self.config["pool_type"] = pool_type
        if shortcut_kernel_size != 1:
            self.config["shortcut_kernel_size"] = shortcut_kernel_size
        if pool_size:
            self.config["pool_size"] = pool_size
        self.pool_size = pool_size
        self.stage_widths = effective_widths
        self.stage_blocks = effective_blocks
        self.kernel_sizes = effective_kernels
        self.effective_depth = 2 * sum(effective_blocks) + 4
        total_blocks = sum(effective_blocks)
        self.stem = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        stages = []
        current_channels = 16
        block_index = 0
        for stage_index, channels in enumerate(effective_widths):
            blocks = []
            for within_stage in range(effective_blocks[stage_index]):
                stride = 2 if stage_index > 0 and within_stage == 0 else 1
                # Linearly increasing rate: the final block uses the configured maximum.
                rate = drop_path * block_index / max(total_blocks - 1, 1)
                blocks.append(
                    PreActBlock(current_channels, channels, stride, dropout, rate, se_ratio,
                                effective_kernels[stage_index], shortcut_type, shortcut_kernel_size)
                )
                current_channels = channels
                block_index += 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.final_bn = nn.BatchNorm2d(current_channels)
        self.final_relu = nn.ReLU(inplace=True)
        if pool_size:
            # CIFAR32 ends in an 8x8 feature map; retain four spatial cells.
            self.pool = nn.AvgPool2d(pool_size, stride=pool_size)
            self.classifier = nn.Linear(current_channels * 4, num_classes)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1) if pool_type == "avg" else (nn.AdaptiveMaxPool2d(1) if pool_type == "max" else AverageMaxPool())
            self.classifier = nn.Linear(current_channels * (2 if pool_type == "avgmax" else 1), num_classes)
        enforce_parameter_budget(self)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.pool_size and (x.ndim != 4 or x.shape[-2:] != (32, 32)):
            raise ValueError("pool_size=4 is defined only for NCHW CIFAR32 inputs")
        x = self.stages(self.stem(x))
        x = self.final_relu(self.final_bn(x))
        return self.classifier(self.pool(x).flatten(1))


def build_model(config: Mapping | None = None) -> WideResNet:
    """Build from model configuration only, always with random initialization."""
    options = dict(config or {})
    allowed = {"depth", "widen_factor", "dropout", "drop_path", "se_ratio", "num_classes",
               "stage_blocks", "stage_widths", "kernel_sizes", "shortcut_type", "pool_type",
               "shortcut_kernel_size", "pool_size"}
    unexpected = options.keys() - allowed
    if unexpected:
        raise ValueError(f"Unknown model option(s): {', '.join(sorted(unexpected))}")
    return WideResNet(**options)


def audit() -> list[dict]:
    """Return measured tensor counts; these are not accuracy estimates."""
    variants = [
        {"depth": 16, "widen_factor": 5.0},
        {"depth": 22, "widen_factor": 4.0},
        {"depth": 28, "widen_factor": 3.5},
        {"depth": 28, "widen_factor": 3.5, "se_ratio": 0.25},
        {"depth": 28, "widen_factor": 3.625, "se_ratio": 0.25},
    ]
    results = []
    for config in variants:
        model = build_model(config)
        count = count_parameters(model)
        results.append({
            **model.config,
            "stage_widths": list(model.stage_widths),
            "total_parameters": count,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "budget_remaining": PARAMETER_LIMIT - count,
        })
    return results


ARCHITECTURE_CANDIDATES = {
    "original": {},
    "blocks_front": {"stage_blocks": [6, 4, 2], "widen_factor": 3.5},
    "blocks_back": {"stage_blocks": [2, 4, 6], "widen_factor": 3.0},
    "blocks_middle": {"stage_blocks": [3, 6, 3], "widen_factor": 3.5},
    "widths_non_doubling": {"stage_widths": [80, 128, 192]},
    "kernel5_all": {"widen_factor": 2.125, "kernel_sizes": [5, 5, 5]},
    "kernel5_early": {"kernel_sizes": [5, 3, 3]},
    "deep40": {"depth": 40, "widen_factor": 2.75},
    "shortcut_avgpool": {"shortcut_type": "avgpool_projection"},
    "shortcut_zero_pad": {"shortcut_type": "zero_pad"},
    "pool_max": {"pool_type": "max"},
    "pool_avgmax": {"pool_type": "avgmax"},
}


def architecture_audit() -> list[dict]:
    """Measure architecture candidates; counts provide no accuracy evidence.

    Kernel/depth redistribution candidates reduce width when needed to fit the
    hard budget. They are feasible architecture comparisons, not parameter-
    matched single-factor ablations. Report these coupled changes explicitly.
    """
    records = []
    for name, config in ARCHITECTURE_CANDIDATES.items():
        model = build_model(config)
        count = count_parameters(model)
        records.append({"name": name, "config": model.config,
                        "stage_blocks": list(model.stage_blocks), "stage_widths": list(model.stage_widths),
                        "kernel_sizes": list(model.kernel_sizes), "effective_depth": model.effective_depth,
                        "total_parameters": count, "budget_remaining": PARAMETER_LIMIT - count})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["audit", "architecture_audit"], default="audit")
    args = parser.parse_args()
    print(json.dumps(architecture_audit() if args.command == "architecture_audit" else audit(), indent=2))


if __name__ == "__main__":
    main()
