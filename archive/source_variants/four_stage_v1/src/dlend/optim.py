"""Mixing, EMA, SAM, and per-update schedules with optional warmup."""
from __future__ import annotations

import copy
import math
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mix_batch(images, labels, num_classes, mixup_alpha=0.0, cutmix_alpha=0.0,
              probability=1.0, smoothing=0.0):
    targets = F.one_hot(labels, num_classes).to(dtype=images.dtype)
    targets = targets * (1 - smoothing) + smoothing / num_classes
    if (mixup_alpha <= 0 and cutmix_alpha <= 0) or np.random.random() >= probability:
        return images, targets
    index = torch.randperm(images.shape[0], device=images.device)
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.random() < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    lam = float(np.random.beta(alpha, alpha))
    if use_cutmix:
        height, width = images.shape[-2:]
        cut_ratio = math.sqrt(1.0 - lam)
        cut_h, cut_w = round(height * cut_ratio), round(width * cut_ratio)
        cy, cx = np.random.randint(height), np.random.randint(width)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + (cut_h + 1) // 2, height)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + (cut_w + 1) // 2, width)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (height * width)
    else:
        images = images * lam + images[index] * (1 - lam)
    return images, targets * lam + targets[index] * (1 - lam)


def classification_loss(logits, targets, teacher_logits=None, alpha=0.5, temperature=4.0):
    supervised = -(targets.float() * F.log_softmax(logits.float(), dim=1)).sum(dim=1).mean()
    if teacher_logits is None:
        return supervised
    kd = F.kl_div(F.log_softmax(logits.float() / temperature, dim=1),
                  F.softmax(teacher_logits.float() / temperature, dim=1),
                  reduction="batchmean") * temperature**2
    return (1 - alpha) * supervised + alpha * kd


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for ema_p, p in zip(self.module.parameters(), model.parameters(), strict=True):
            ema_p.lerp_(p.detach(), 1 - decay)
        # BN statistics are copied, not averaged across incompatible time horizons.
        for ema_b, b in zip(self.module.buffers(), model.buffers(), strict=True):
            ema_b.copy_(b)


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(groups, lr=cfg["lr"], momentum=cfg["momentum"], nesterov=cfg["nesterov"])
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, lr=cfg["lr"])
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(groups, lr=cfg["lr"])
    if cfg["optimizer"] == "rmsprop":
        return torch.optim.RMSprop(groups, lr=cfg["lr"], momentum=cfg["momentum"])
    raise ValueError(f"Unknown optimizer {cfg['optimizer']}")


def learning_rate(step, total_steps, warmup_steps, maximum, minimum, scheduler="cosine"):
    """Warm up, then decay over [0, 1] of the remaining update interval.

    Multistep multiplies by 0.1 at remaining-progress 0.5 and 0.75; minimum
    is a lower bound, not a forced final value. The default cosine expression
    deliberately retains the exact historical operation order.
    """
    if scheduler not in {"cosine", "linear", "multistep"}:
        raise ValueError(f"Unknown scheduler {scheduler}")
    if warmup_steps and step < warmup_steps:
        return maximum * (step + 1) / warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)))
    if scheduler == "linear":
        return minimum + (maximum - minimum) * (1 - progress)
    if scheduler == "multistep":
        drops = sum(progress >= milestone for milestone in (0.5, 0.75))
        return max(minimum, maximum * 0.1**drops)
    return minimum + (maximum - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def sam_perturb(model, rho):
    parameters = [p for p in model.parameters() if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm(2) for p in parameters]))
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite first-pass SAM gradient")
    saved = []
    for p in parameters:
        saved.append((p, p.detach().clone()))
        p.add_(p.grad, alpha=float(rho / (norm + 1e-12)))
    return saved


@torch.no_grad()
def sam_restore(saved):
    for p, original in saved:
        p.copy_(original)


@contextmanager
def freeze_bn_statistics(model):
    modules = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    values = [(m, m.momentum, m.num_batches_tracked.clone() if m.num_batches_tracked is not None else None)
              for m in modules]
    try:
        for m in modules:
            m.momentum = 0.0
        yield
    finally:
        for m, momentum, tracked in values:
            m.momentum = momentum
            if tracked is not None:
                m.num_batches_tracked.copy_(tracked)
