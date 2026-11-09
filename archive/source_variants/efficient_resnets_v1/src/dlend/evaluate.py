"""Evaluate the saved, fingerprint-verified validation split (never label it test)."""
from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import build_datasets
from .models import build_model, count_parameters
from .train import evaluate_loader
from .utils import get_device, load_checkpoint, write_json


def wilson_interval(correct, total, z=1.96):
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    delta = z * math.sqrt(p * (1-p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center-delta), min(1.0, center+delta)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", help="Relocate the same data; fingerprints must still match")
    parser.add_argument("--tta", choices=["none", "flip", "fivecrop"], default="none")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--weights", choices=["model", "ema"], default="model")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid batch-size or workers")
    checkpoint = load_checkpoint(args.checkpoint)
    config = copy.deepcopy(checkpoint["config"])
    if args.data_root:
        config["dataset"]["root"] = args.data_root
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="eval-split-") as scratch:
        _, validation, metadata = build_datasets(config, Path(scratch))
    if validation is None:
        raise ValueError("Checkpoint used full-data refit: there is no independent validation split")
    for key in ("fingerprint", "split_fingerprint", "class_names"):
        if metadata[key] != checkpoint["metadata"][key]:
            raise ValueError(f"Validation data changed: {key}")
    device = get_device(args.device)
    torch.set_num_threads(config["training"].get("threads", 4))
    model = build_model(config["model"]).to(device)
    state = checkpoint.get(args.weights)
    if not isinstance(state, dict):
        raise ValueError(f"No {args.weights} weights in checkpoint")
    model.load_state_dict(state)
    loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    result = evaluate_loader(model, loader, device, config["training"].get("amp", "none"), tta=args.tta)
    matrix = [[0] * 10 for _ in range(10)]
    for target, pred in zip(result.pop("targets"), result.pop("predictions"), strict=True):
        matrix[target][pred] += 1
    correct = sum(matrix[i][i] for i in range(10))
    result.update(split="validation", tta=args.tta, parameters=count_parameters(model),
                  synthetic=metadata["synthetic"], confusion_matrix=matrix, class_names=metadata["class_names"],
                  wilson_95_interval=wilson_interval(correct, result["count"]),
                  interval_note="Descriptive only; validation model selection and duplicate groups violate a pristine independent test assumption.",
                  checkpoint=str(Path(args.checkpoint).resolve()), weights=args.weights,
                  split_fingerprint=metadata["split_fingerprint"])
    write_json(result, output)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
