"""Generate a single-model competition CSV in exact sample-submission order."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ImageDataset, load_csv_records, make_transform, read_csv_rows
from .models import build_model, count_parameters


@torch.inference_mode()
def predict_probabilities(model, images, tta="none"):
    """Average class probabilities over views of the same model (no ensemble)."""
    if tta == "none":
        views = [images]
    elif tta == "flip":
        views = [images, images.flip(-1)]
    elif tta == "fivecrop":
        padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
        views = [padded[:, :, y:y + 32, x:x + 32] for y, x in [(0, 0), (0, 8), (8, 0), (8, 8), (4, 4)]]
    else:
        raise ValueError(f"Unknown TTA: {tta}")
    return torch.stack([model(view).float().softmax(-1) for view in views]).mean(0)


def ordered_submission(sample_path, predictions, class_names, output_path, id_column="id", label_column="label"):
    """Validate bijection with sample IDs before writing any submission bytes."""
    rows, columns = read_csv_rows(Path(sample_path))
    if id_column not in columns or label_column not in columns:
        raise ValueError(f"Sample must contain explicit ID/label columns: {id_column}, {label_column}; got {columns}")
    sample_ids = [row[id_column] for row in rows]
    if len(set(sample_ids)) != len(sample_ids) or any(not item.strip() for item in sample_ids):
        raise ValueError("Sample submission contains duplicate or empty IDs")
    if set(sample_ids) != set(predictions):
        missing = sorted(set(sample_ids) - set(predictions))[:5]
        extra = sorted(set(predictions) - set(sample_ids))[:5]
        raise ValueError(f"Sample/test ID mismatch; missing predictions={missing}, extra predictions={extra}")
    classes = [str(value) for value in class_names]
    if not classes or len(classes) != len(set(classes)) or any(not value.strip() for value in classes):
        raise ValueError("Class names must be non-empty and unique")
    for row in rows:
        index = predictions[row[id_column]]
        if not isinstance(index, int) or not 0 <= index < len(classes):
            raise ValueError(f"Invalid predicted class index {index}")
        row[label_column] = classes[index]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", required=True, help="Manifest with ID and optional image path column")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--sample-submission", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--path-column")
    parser.add_argument("--image-extension")
    parser.add_argument("--tta", choices=["none", "flip", "fivecrop"], default="none")
    parser.add_argument("--weights", choices=["model", "ema"], default="model", help="best.pt model already contains the selected weights")
    parser.add_argument("--class-names-json", help="Explicit output labels in checkpoint class-index order, only if competition encoding differs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-synthetic", action="store_true", help="Debug only: permit a synthetic smoke checkpoint")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers non-negative")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata.get("synthetic") and not args.allow_synthetic:
        raise ValueError("Synthetic smoke checkpoint cannot produce a competition submission; use a real-data checkpoint")
    classes = metadata["class_names"]
    if args.class_names_json:
        mapped_classes = json.loads(Path(args.class_names_json).read_text(encoding="utf-8"))
        if not isinstance(mapped_classes, list) or len(mapped_classes) != len(classes) or any(not isinstance(v, (str, int)) for v in mapped_classes):
            raise ValueError("class-names-json must contain one string/integer per model class, in trained index order")
        classes = mapped_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model = build_model(checkpoint["config"]["model"])
    weights = checkpoint.get(args.weights)
    if not isinstance(weights, dict):
        raise ValueError(f"Checkpoint has no {args.weights} state dictionary")
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    cfg = {"test_csv": args.test_csv, "image_dir": args.image_dir, "id_column": args.id_column, "path_column": args.path_column, "image_extension": args.image_extension}
    records = load_csv_records(cfg, labeled=False)
    dataset = ImageDataset(records, make_transform({}, False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    predictions = {}
    for images, ids in loader:
        probabilities = predict_probabilities(model, images.to(device, non_blocking=True), args.tta)
        if not torch.isfinite(probabilities).all():
            raise ValueError("Model produced non-finite probabilities; no submission was written")
        if probabilities.shape[1] != len(classes):
            raise ValueError("Checkpoint class-name count differs from model outputs")
        for item_id, index in zip(ids, probabilities.argmax(1).cpu().tolist()):
            predictions[item_id] = index
    count = ordered_submission(args.sample_submission, predictions, classes, args.output, args.id_column, args.label_column)
    provenance = {"checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_type": checkpoint.get("checkpoint_type"), "selected_source": checkpoint.get("selected_source"), "weights": args.weights, "parameters": count_parameters(model), "tta": args.tta, "rows": count, "class_names": classes, "training_fingerprint": metadata.get("fingerprint"), "synthetic": bool(metadata.get("synthetic")), "test_ids": [r["id"] for r in records]}
    Path(str(args.output) + ".metadata.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(Path(args.output).resolve()), "rows": count, "parameters": count_parameters(model), "tta": args.tta, "synthetic": bool(metadata.get("synthetic"))}))


if __name__ == "__main__":
    main()
