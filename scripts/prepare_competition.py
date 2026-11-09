"""Audit the supplied Kaggle archive and prepare NPZ training + test images.

Images are decoded directly from ZIP; only test images and the three original
CSV files are extracted. No external data or test labels are obtained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import stat
import uuid
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_members(archive):
    """Reject traversal, links, duplicated archive names, and oversized members."""
    names = set()
    for member in archive.infolist():
        name = member.filename
        path = PurePosixPath(name)
        parts = name.rstrip("/").split("/")
        if not name or "\\" in name or path.is_absolute() or any(part in {"", ".", ".."} or ":" in part for part in parts):
            raise ValueError(f"Unsafe ZIP member path: {name!r}")
        if name in names:
            raise ValueError(f"Duplicate ZIP member: {name}")
        names.add(name)
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(f"ZIP symbolic links are not supported: {name}")
        if member.file_size > 50_000_000:
            raise ValueError(f"Unexpected oversized ZIP member: {name}")
    return names


def read_manifest(archive, name, columns):
    payload = archive.read(name)
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != columns:
        raise ValueError(f"Expected {name} columns {columns}, got {reader.fieldnames}")
    rows = list(reader)
    seen_ids, seen_files = set(), set()
    for row in rows:
        if None in row or any(value is None or not value.strip() for value in row.values()):
            raise ValueError(f"Missing/malformed value in {name}")
        item_id = row["id"]
        if item_id in seen_ids:
            raise ValueError(f"Duplicate ID in {name}: {item_id}")
        seen_ids.add(item_id)
        if "filename" in row:
            filename = row["filename"]
            if Path(filename).name != filename or "/" in filename or "\\" in filename or ":" in filename:
                raise ValueError(f"Expected a basename in {name}, got {filename}")
            if filename.casefold() in seen_files:
                raise ValueError(f"Duplicate filename in {name}: {filename}")
            seen_files.add(filename.casefold())
    return rows, payload


def decode_image(payload, filename):
    with Image.open(io.BytesIO(payload)) as image:
        if image.size != (32, 32) or image.mode != "RGB":
            raise ValueError(f"Expected 32x32 RGB image, got {image.size}/{image.mode}: {filename}")
        image.load()
        return np.array(image, dtype=np.uint8, copy=True)


def prepare_archive(archive_path, output, expected_train=50_000, expected_test=2_000, expected_classes=10):
    source = Path(archive_path).resolve()
    output = Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be absent or empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}-preparing-{uuid.uuid4().hex[:12]}"
    stage.mkdir()
    archive_sha = sha256_file(source)
    print(json.dumps({"stage": "archive_audit", "bytes": source.stat().st_size, "sha256": archive_sha}), flush=True)
    with zipfile.ZipFile(source) as archive:
        names = audit_members(archive)
        train_rows, train_csv = read_manifest(archive, "train_labels.csv", ["id", "filename", "label"])
        test_rows, test_csv = read_manifest(archive, "test.csv", ["id", "filename"])
        sample_rows, sample_csv = read_manifest(archive, "sample_submission.csv", ["id", "label"])
        if len(train_rows) != expected_train or len(test_rows) != expected_test:
            raise ValueError(f"Unexpected train/test counts: {len(train_rows)}/{len(test_rows)}")
        if [row["id"] for row in sample_rows] != [row["id"] for row in test_rows]:
            raise ValueError("sample_submission.csv must contain exactly the test.csv IDs in the same order")
        classes = [str(index) for index in range(expected_classes)]
        counts = Counter(row["label"] for row in train_rows)
        if set(counts) != set(classes) or any(counts[label] * expected_classes != expected_train for label in classes):
            raise ValueError(f"Expected balanced classes 0..{expected_classes-1}; got {dict(counts)}")
        expected_files = {"train_labels.csv", "test.csv", "sample_submission.csv"}
        expected_files.update("train_images/" + row["filename"] for row in train_rows)
        expected_files.update("test_images/" + row["filename"] for row in test_rows)
        actual_files = {name for name in names if not name.endswith("/")}
        if actual_files != expected_files:
            raise ValueError(f"Archive/CSV file mismatch: missing={sorted(expected_files-actual_files)[:5]}, extra={sorted(actual_files-expected_files)[:5]}")
        images = np.empty((len(train_rows), 32, 32, 3), dtype=np.uint8)
        labels = np.array([int(row["label"]) for row in train_rows], dtype=np.int64)
        ids = np.array([row["id"] for row in train_rows], dtype=str)
        train_hashes = defaultdict(list)
        for index, row in enumerate(train_rows):
            name = "train_images/" + row["filename"]
            pixels = decode_image(archive.read(name), name)
            images[index] = pixels
            train_hashes[hashlib.sha256(pixels.tobytes()).hexdigest()].append(index)
            if (index + 1) % 10_000 == 0:
                print(json.dumps({"stage": "decode_train", "completed": index + 1, "total": len(train_rows)}), flush=True)
        with (stage / "train.npz").open("wb") as stream:
            np.savez(stream, images=images, labels=labels, ids=ids)
        test_hashes, overlap = defaultdict(list), []
        (stage / "test_images").mkdir()
        for index, row in enumerate(test_rows):
            name = "test_images/" + row["filename"]
            payload = archive.read(name)
            pixels = decode_image(payload, name)
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            test_hashes[digest].append(index)
            if digest in train_hashes:
                overlap.append({"test_id": row["id"], "train_ids": [train_rows[i]["id"] for i in train_hashes[digest]]})
            (stage / "test_images" / row["filename"]).write_bytes(payload)
        for name, payload in [("train_labels.csv", train_csv), ("test.csv", test_csv), ("sample_submission.csv", sample_csv)]:
            (stage / name).write_bytes(payload)
        contradictory = [{"train_ids": [train_rows[i]["id"] for i in group]} for group in train_hashes.values() if len({train_rows[i]["label"] for i in group}) > 1]
        audit = {
            "source_archive": str(source), "source_sha256": archive_sha, "source_bytes": source.stat().st_size,
            "archive_entries": len(names), "train_count": len(train_rows), "test_count": len(test_rows),
            "class_names": classes, "class_counts": dict(sorted(counts.items())),
            "image_shape": [32, 32, 3], "image_dtype": "uint8", "label_dtype": "int64",
            "train_id_order": "exact train_labels.csv order", "test_id_order": "exact test.csv/sample_submission.csv order",
            "train_duplicate_pixel_groups": sum(len(group) > 1 for group in train_hashes.values()),
            "test_duplicate_pixel_groups": sum(len(group) > 1 for group in test_hashes.values()),
            "train_conflicting_label_groups": contradictory,
            "train_test_exact_pixel_overlap_count": len(overlap), "train_test_exact_pixel_overlaps": overlap,
            "test_labels_available": False, "sample_submission_labels": "placeholders; never used for training or evaluation",
            "csv_sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in [("train_labels.csv", train_csv), ("test.csv", test_csv), ("sample_submission.csv", sample_csv)]},
            "train_npz_sha256": sha256_file(stage / "train.npz"),
            "provenance": "Only the user-supplied competition archive was read; no external datasets or labels were obtained.",
        }
    (stage / "preparation_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if output.exists():
        output.rmdir()  # Earlier check requires an empty output directory.
    stage.rename(output)
    print(json.dumps({"stage": "complete", "output": str(output), "train": len(train_rows), "test": len(test_rows), "train_test_overlap": len(overlap), "conflicting_label_groups": len(contradictory)}), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare_archive(args.archive, args.output)


if __name__ == "__main__":
    main()
