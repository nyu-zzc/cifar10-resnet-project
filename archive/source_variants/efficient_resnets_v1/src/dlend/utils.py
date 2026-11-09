"""Reproducibility, safe checkpoint IO, and command line configuration."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def rng_state() -> dict:
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])


def atomic_save(value, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def write_json(value, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu") -> dict:
    # Only tensor/primitive containers; never enable arbitrary pickle execution.
    return torch.load(path, map_location=device, weights_only=True)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Expected dotted.key=value: {item}")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        target = config
        keys = key.split(".")
        for part in keys[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"Not an object: {key}")
        target[keys[-1]] = value
    return config


def get_device(request: str = "auto") -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if request == "auto" else torch.device(request)


def environment() -> dict:
    import torchvision
    return {
        "python": sys.version, "platform": platform.platform(),
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "numpy": str(np.__version__), "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
