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


class AuthorLookaheadSGD(torch.optim.SGD):
    """Modern SGD preserving the supplied Lookahead steps 1, 6, 11, ...

    Step 1 initializes slow weights after the fast update (a no-op blend).
    Momentum is never reset. Slow tensors share Optimizer.state with momentum,
    giving stable parameter-index serialization instead of Python object ids.
    """
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0, nesterov=False, k=5, alpha=0.5):
        if type(k) is not int or k < 1:
            raise ValueError("Lookahead k must be a positive integer")
        if not isinstance(alpha, (float, int)) or isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Lookahead alpha must be finite and in [0,1]")
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        self.defaults.update(lookahead_k=k, lookahead_alpha=float(alpha), lookahead_counter=0)
        for group in self.param_groups:
            group.update(lookahead_k=k, lookahead_alpha=float(alpha), lookahead_counter=0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for group in self.param_groups:
            if group["lookahead_counter"] == 0:
                for parameter in group["params"]:
                    state = self.state[parameter]
                    if "slow_param" not in state:
                        state["slow_param"] = parameter.detach().clone()
                    slow = state["slow_param"]
                    slow.add_((parameter - slow) * group["lookahead_alpha"])
                    parameter.copy_(slow)
            group["lookahead_counter"] = (group["lookahead_counter"] + 1) % group["lookahead_k"]
        return loss

    def load_state_dict(self, state_dict):
        for group in state_dict["param_groups"]:
            k, alpha, counter = (group.get(key) for key in ("lookahead_k", "lookahead_alpha", "lookahead_counter"))
            if k != self.defaults["lookahead_k"] or alpha != self.defaults["lookahead_alpha"]:
                raise ValueError("Lookahead checkpoint hyperparameters differ")
            if type(counter) is not int or not 0 <= counter < k:
                raise ValueError("Invalid Lookahead checkpoint counter")
        super().load_state_dict(state_dict)


def make_optimizer(model, cfg):
    # Author main.py decays every trainable tensor, including BN and biases.
    groups = [{"params": [p for p in model.parameters() if p.requires_grad],
               "weight_decay": cfg["weight_decay"]}]
    if cfg["optimizer"] == "sgd":
        if cfg.get("lookahead_k", 0):
            return AuthorLookaheadSGD(groups, lr=cfg["lr"], momentum=cfg["momentum"],
                                     nesterov=cfg["nesterov"], weight_decay=cfg["weight_decay"],
                                     k=cfg["lookahead_k"], alpha=cfg["lookahead_alpha"])
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
