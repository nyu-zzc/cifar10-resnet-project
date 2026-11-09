"""Local competition data, deterministic splits, and CIFAR training transforms.

No dataset is downloaded. CSV labels are interpreted as exact strings so the
saved class order also defines the submission encoding.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class Cutout:
    """Mask one square in a normalized training image."""
    def __init__(self, length: int):
        if length < 0 or length > 32:
            raise ValueError("cutout must be between 0 and 32")
        self.length = length

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.length == 0:
            return tensor
        h, w = tensor.shape[-2:]
        y, x = int(torch.randint(h, ()).item()), int(torch.randint(w, ()).item())
        y0, x0 = max(0, y - self.length // 2), max(0, x - self.length // 2)
        y1, x1 = min(h, y + (self.length + 1) // 2), min(w, x + (self.length + 1) // 2)
        tensor = tensor.clone()
        tensor[:, y0:y1, x0:x1] = 0
        return tensor


def make_transform(config: dict[str, Any], train: bool = False):
    steps: list[Any] = []
    if train:
        steps.extend([T.RandomCrop(32, padding=4, padding_mode="reflect"), T.RandomHorizontalFlip()])
        augment = config.get("augment", "basic")
        if augment == "randaugment":
            steps.append(T.RandAugment(num_ops=int(config.get("ra_num_ops", 2)), magnitude=int(config.get("ra_magnitude", 7))))
        elif augment == "autoaugment":
            steps.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
        elif augment == "augmix":
            # Single-view torchvision variant; no multi-view/JSD loss is added.
            severity = config.get("augmix_severity", 3)
            width = config.get("augmix_mixture_width", 3)
            depth = config.get("augmix_chain_depth", -1)
            for key, value in (("augmix_severity", severity),
                               ("augmix_mixture_width", width),
                               ("augmix_chain_depth", depth)):
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(f"{key} must be an integer")
            if not 1 <= severity <= 10:
                raise ValueError("augmix_severity must be in [1, 10]")
            if width < 1:
                raise ValueError("augmix_mixture_width must be positive")
            if depth != -1 and depth < 1:
                raise ValueError("augmix_chain_depth must be -1 (random 1..3) or positive")
            alpha = config.get("augmix_alpha", 1.0)
            if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
                raise ValueError("augmix_alpha must be a positive finite number")
            all_ops = config.get("augmix_all_ops", False)
            if not isinstance(all_ops, bool):
                raise ValueError("augmix_all_ops must be a boolean")
            steps.append(T.AugMix(severity=severity, mixture_width=width,
                                  chain_depth=depth, alpha=float(alpha), all_ops=all_ops,
                                  interpolation=T.InterpolationMode.BILINEAR, fill=None))
        elif augment != "basic":
            raise ValueError(f"Unknown augment: {augment}")
        # Opt-in appearance changes follow the existing augmentation policy.
        # Omitted/zero values append no transform and consume no extra RNG draws.
        appearance = {}
        for key in ("color_jitter", "grayscale_probability", "gaussian_blur_probability"):
            try:
                value = float(config.get(key, 0))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{key} must be a finite number in [0, 1]") from error
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be a finite number in [0, 1]")
            appearance[key] = value
        jitter = appearance["color_jitter"]
        if jitter:
            steps.append(T.ColorJitter(brightness=jitter, contrast=jitter,
                                       saturation=jitter, hue=min(0.05, jitter * 0.25)))
        if appearance["grayscale_probability"]:
            steps.append(T.RandomGrayscale(p=appearance["grayscale_probability"]))
        if appearance["gaussian_blur_probability"]:
            steps.append(T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                                       p=appearance["gaussian_blur_probability"]))
    steps.extend([T.ToTensor(), T.Normalize(MEAN, STD)])
    if train and config.get("cutout", 0):
        steps.append(Cutout(int(config["cutout"])))
    if train and config.get("random_erasing", 0):
        probability = float(config["random_erasing"])
        if not 0 <= probability <= 1:
            raise ValueError("random_erasing must be in [0, 1]")
        steps.append(T.RandomErasing(p=probability, scale=(0.02, 0.2), ratio=(0.3, 3.3)))
    return T.Compose(steps)


def _rooted(root: Path, value: str | Path) -> Path:
    value = Path(value)
    return (value if value.is_absolute() else root / value).resolve()


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if not columns or len(columns) != len(set(columns)):
            raise ValueError(f"CSV must have unique column names: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row: {path}")
    return rows, columns


def load_csv_records(config: dict[str, Any], *, labeled: bool = True):
    """Read a CSV manifest without sorting IDs or changing their representation."""
    root = Path(config.get("root", ".")).resolve()
    csv_key = "train_csv" if labeled else "test_csv"
    if not config.get(csv_key):
        raise ValueError(f"Supply {csv_key}; no competition data are bundled or downloaded")
    csv_path = _rooted(root, config[csv_key])
    image_root = _rooted(root, config.get("image_dir", "."))
    rows, columns = read_csv_rows(csv_path)
    id_col = config.get("id_column", "id")
    label_col = config.get("label_column", "label")
    path_col = config.get("path_column")
    required = [id_col] + ([label_col] if labeled else []) + ([path_col] if path_col else [])
    if any(column not in columns for column in required):
        raise ValueError(f"CSV missing required columns {required}; got {columns}")
    records, seen_ids, seen_paths = [], set(), set()
    for row in rows:
        item_id = row[id_col]
        if not item_id.strip() or item_id in seen_ids:
            raise ValueError(f"Empty or duplicate ID: {item_id!r}")
        seen_ids.add(item_id)
        relative = row[path_col] if path_col else item_id
        if not relative.strip():
            raise ValueError(f"Missing image path for ID {item_id}")
        image_path = _rooted(image_root, relative)
        if not image_path.is_file() and not Path(relative).suffix:
            extensions = [config["image_extension"]] if config.get("image_extension") else [".png", ".jpg", ".jpeg", ".bmp"]
            candidates = [Path(str(image_path) + (e if str(e).startswith(".") else "." + e)) for e in extensions]
            candidates = [p for p in candidates if p.is_file()]
            if len(candidates) != 1:
                raise ValueError(f"Expected one image for ID {item_id}; found {len(candidates)}. Set path_column or image_extension.")
            image_path = candidates[0]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        path_key = os.path.normcase(str(image_path))
        if path_key in seen_paths:
            raise ValueError(f"Duplicate image path: {image_path}")
        seen_paths.add(path_key)
        label = row[label_col] if labeled else None
        if labeled and not label.strip():
            raise ValueError(f"Missing label for ID {item_id}")
        records.append({"id": item_id, "path": str(image_path), "label": label})
    return records


class ImageDataset(Dataset):
    def __init__(self, records, transform, class_names=None, arrays=None):
        self.records = records
        self.transform = transform
        self.arrays = arrays
        self.class_names = list(class_names or [])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.targets = [self.class_to_idx[r["label"]] for r in records] if self.class_names else []

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if self.arrays is None:
            with Image.open(record["path"]) as source:
                image = source.convert("RGB")
        else:
            image = Image.fromarray(self.arrays[record["array_index"]])
        if image.size != (32, 32):
            raise ValueError(f"Expected CIFAR image 32x32, got {image.size} for {record['id']}")
        x = self.transform(image)
        return (x, self.targets[index]) if self.class_names else (x, record["id"])


def _class_order(labels, explicit=None):
    unique = set(labels)
    if explicit is not None:
        classes = [str(value) for value in explicit]
        if len(classes) != len(set(classes)) or set(classes) != unique:
            raise ValueError("class_names must list every observed label exactly once")
        return classes
    try:
        return sorted(unique, key=lambda x: (int(x), x))
    except ValueError:
        return sorted(unique)


def _scalar_string(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _load_npz(path: Path):
    with np.load(path, allow_pickle=False) as source:
        if "images" not in source or "labels" not in source:
            raise ValueError("Training NPZ requires images and labels arrays")
        images, labels = source["images"], source["labels"]
        ids = source["ids"] if "ids" in source else np.arange(len(images))
    if images.ndim == 4 and images.shape[1:] == (3, 32, 32):
        images = images.transpose(0, 2, 3, 1)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError("NPZ images must be uint8 NHWC or NCHW with 32x32 RGB images")
    if labels.shape != (len(images),) or ids.shape != (len(images),):
        raise ValueError("NPZ labels and ids must each have shape (N,)")
    records = [{"id": _scalar_string(ids[i]), "label": _scalar_string(labels[i]), "array_index": i} for i in range(len(images))]
    if len({r["id"] for r in records}) != len(records) or any(not r["id"].strip() or not r["label"].strip() or r["label"].lower() == "nan" for r in records):
        raise ValueError("NPZ has duplicate IDs or missing labels")
    return records, np.ascontiguousarray(images)


def build_datasets(config: dict[str, Any], output_dir: Path):
    """Return train, validation (or None), metadata; persist exact split IDs.

    A reused output directory must refer to the same bytes, class mapping, seed,
    and split. The fingerprint includes image bytes, labels, and IDs.
    """
    cfg = dict(config.get("dataset", config))
    seed = int(cfg.get("seed", config.get("seed", 42)))
    fraction = float(cfg.get("val_fraction", 0.1))
    if not 0 <= fraction < 1:
        raise ValueError("val_fraction must be in [0,1)")
    arrays = None
    synthetic = bool(cfg.get("synthetic", False))
    if synthetic:
        n, n_classes = int(cfg.get("synthetic_size", 100)), int(cfg.get("num_classes", 10))
        if n_classes < 2 or n < n_classes * 2:
            raise ValueError("Synthetic smoke data require at least two examples per class")
        arrays = np.random.default_rng(seed).integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)
        records = [{"id": f"synthetic-{i:06d}", "label": str(i % n_classes), "array_index": i} for i in range(n)]
    elif cfg.get("npz"):
        records, arrays = _load_npz(_rooted(Path(cfg.get("root", ".")), cfg["npz"]))
    else:
        records = load_csv_records(cfg)
    classes = _class_order([r["label"] for r in records], cfg.get("class_names"))
    if len(classes) != int(cfg.get("num_classes", 10)):
        raise ValueError(f"Expected {cfg.get('num_classes', 10)} classes, observed {len(classes)}; set num_classes explicitly if appropriate")
    digest = hashlib.sha256()
    content_groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        digest.update(json.dumps([record["id"], record["label"]], ensure_ascii=False).encode("utf-8"))
        if arrays is None:
            with Path(record["path"]).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            with Image.open(record["path"]) as source:
                source = source.convert("RGB")
                if source.size != (32, 32):
                    raise ValueError(f"Expected CIFAR image 32x32, got {source.size} for {record['id']}")
                pixel_bytes = source.tobytes()
        else:
            pixel_bytes = arrays[record["array_index"]].tobytes()
            digest.update(pixel_bytes)
        content_hash = hashlib.sha256(pixel_bytes).hexdigest()
        members = content_groups.setdefault(content_hash, [])
        if members and records[members[0]]["label"] != record["label"]:
            raise ValueError(f"Identical image pixels have conflicting labels: {record['id']}")
        members.append(index)
        record["content_hash"] = content_hash
    fingerprint = digest.hexdigest()
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    for label in classes:
        groups = [group for group in content_groups.values() if records[group[0]]["label"] == label]
        if fraction and len(groups) < 2:
            raise ValueError(f"At least two distinct images required in class {label!r} for stratified validation")
        groups = [groups[i] for i in rng.permutation(len(groups))]
        target = max(1, int(round(sum(map(len, groups)) * fraction))) if fraction else 0
        assigned = 0
        for position, group in enumerate(groups):
            if assigned < target and position < len(groups) - 1:
                val_indices.extend(group)
                assigned += len(group)
            else:
                train_indices.extend(group)
    train_indices, val_indices = sorted(train_indices), sorted(val_indices)
    split = {"seed": seed, "val_fraction": fraction, "train_ids": [records[i]["id"] for i in train_indices], "val_ids": [records[i]["id"] for i in val_indices]}
    split_fingerprint = hashlib.sha256(json.dumps(split, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    metadata = {"class_names": classes, "fingerprint": fingerprint, "split_fingerprint": split_fingerprint, "train_size": len(train_indices), "val_size": len(val_indices), "split": split, "synthetic": synthetic, "duplicate_content_groups": sum(len(group) > 1 for group in content_groups.values())}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "split.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Existing split.json differs from current data/config; use a new output directory")
    manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    train = ImageDataset([records[i] for i in train_indices], make_transform(cfg, True), classes, arrays)
    val = ImageDataset([records[i] for i in val_indices], make_transform(cfg, False), classes, arrays) if val_indices else None
    return train, val, metadata
