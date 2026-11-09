"""Explicit, hash-bound admission of an authorized original project SWA artifact."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from .models import count_parameters
from .utils import file_sha256, load_checkpoint

AUTHORIZED_PLAN_SHA = "f32ca4a6a6f06df1b0b5e228c8549daf71810f6b697fc7eaa40a3c3c4037862c"
AUTHORIZED_SWA_SHA = "415cf7e902972f269dc7b385891f6443701525703df34212752a3ae959a4c9b1"


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stable_load(path, expected=None):
    path = Path(path).resolve()
    before = file_sha256(path)
    if expected is not None and before != expected:
        raise ValueError("Registered checkpoint SHA mismatch")
    checkpoint = load_checkpoint(path, device="cpu")
    if file_sha256(path) != before:
        raise ValueError("Checkpoint changed during CPU loading")
    return checkpoint


def verify_state(state):
    if not isinstance(state, dict) or not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Require a complete tensor state dictionary")
    if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError("Nonfinite source model or BN values")
    counters = [v for k, v in state.items() if k.endswith("num_batches_tracked")]
    if not counters or any(v.numel() != 1 or int(v) <= 0 for v in counters):
        raise ValueError("All BN counters must show real training/calibration")
    if any((v < 0).any() for k, v in state.items() if k.endswith("running_var")):
        raise ValueError("Invalid BN running variance")


def verify_ancestry(checkpoint, metadata, seen=None):
    seen = set() if seen is None else seen
    provenance = checkpoint.get("provenance", {})
    if (provenance.get("external_data") is not False or provenance.get("teacher_checkpoint") is not None
            or provenance.get("teacher_sha256") is not None):
        raise ValueError("Only project random ancestry without external data/teacher is admitted")
    source_meta = checkpoint.get("metadata", {})
    for key in ("fingerprint", "split_fingerprint", "class_names", "synthetic", "train_size", "val_size"):
        if key not in metadata or source_meta.get(key) != metadata[key]:
            raise ValueError("Ancestry dataset mismatch: " + key)
    if provenance.get("initialization") == "random":
        if provenance.get("warm_start") is not None:
            raise ValueError("Random provenance cannot contain a warm start")
        return {}
    warm = provenance.get("warm_start", {})
    if provenance.get("initialization") != "warm_start" or not isinstance(warm, dict):
        raise ValueError("Missing project random ancestry")
    path, digest = Path(warm.get("checkpoint", "")).resolve(), warm.get("sha256")
    if str(path) in seen or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Cyclic or incomplete ancestry")
    seen.add(str(path))
    parent = stable_load(path, digest)
    if parent.get("provenance") != warm.get("source_provenance"):
        raise ValueError("Parent provenance changed")
    source_total = parent.get("trained_epochs_total", parent.get("epoch"))
    if warm.get("trained_epochs_total") != source_total:
        raise ValueError("Parent training lineage mismatch")
    return {str(path): digest, **verify_ancestry(parent, metadata, seen)}


def load_registered_swa(path, model, metadata, registration_path, registration_sha256):
    """No inference and no copied optimizer/EMA/SWA accumulator or RNG state."""
    from .train import resolve_config

    path, registration_path = Path(path).resolve(), Path(registration_path).resolve()
    if (not isinstance(registration_sha256, str) or len(registration_sha256) != 64
            or file_sha256(registration_path) != registration_sha256):
        raise ValueError("Explicit artifact registration SHA is required and must match")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("schema_version") != 1 or registration.get("kind") != "registered_project_swa_warm_start":
        raise ValueError("Unsupported artifact registration")
    source = registration["source"]
    if (path.name != "swa.pt" or path != Path(source["checkpoint"]).resolve() or source["artifact_type"] != "swa"
            or source["checkpoint_sha256"] != AUTHORIZED_SWA_SHA):
        raise ValueError("Must load the original registered swa.pt, never a renamed alias")
    required = {str(path.with_name(f)) for f in ("swa.pt", "swa.json", "best.pt", "best.json", "config.json",
                "dataset.json", "split.json", "metrics.jsonl", "last.pt", "final.pt")}
    required.update([source["campaign"], source["completion_log"]])
    if not required.issubset(source["files_sha256"]):
        raise ValueError("Registration omits required source evidence")
    for filename, expected in source["files_sha256"].items():
        if file_sha256(filename) != expected:
            raise ValueError("Registered source evidence changed: " + filename)
    authorization = registration["authorization"]
    if (authorization["plan_sha256"] != AUTHORIZED_PLAN_SHA
            or file_sha256(authorization["plan"]) != authorization["plan_sha256"]):
        raise ValueError("Authorization plan changed")
    plan = json.loads(Path(authorization["plan"]).read_text(encoding="utf-8"))
    matches = [j for j in plan["jobs"] if j["name"] == authorization["job"]]
    if len(matches) != 1:
        raise ValueError("Artifact has no single explicit authorized job")
    approved = matches[0]
    if (approved["source_artifact_type"] != "swa" or Path(approved["source_checkpoint"]).resolve() != path
            or approved["source_checkpoint_sha256"] != source["checkpoint_sha256"]
            or approved["parameters"] != source["parameters"]
            or approved["source_trained_epochs_total"] != source["trained_epochs_total"]):
        raise ValueError("SWA source differs from approved job")
    checkpoint = stable_load(path, source["checkpoint_sha256"])
    canonical = stable_load(path.with_name("best.pt"), source["canonical_best_sha256"])
    run_config = json.loads(path.with_name("config.json").read_text(encoding="utf-8"))
    run_meta = json.loads(path.with_name("dataset.json").read_text(encoding="utf-8"))
    campaign = json.loads(Path(source["campaign"]).read_text(encoding="utf-8"))
    entry = campaign["jobs"][source["entry_name"]]
    if (campaign["status"] not in {"complete", "converged"} or entry["status"] != "complete"
            or entry["phase"] != "full" or json_sha(entry) != source["entry_sha256"]
            or Path(entry["run_dir"]).resolve() != path.parent
            or Path(entry["checkpoint"]).resolve() != path.with_name("best.pt")):
        raise ValueError("Require the registered completed full source run")
    epochs = run_config["training"]["epochs"]
    rows = [json.loads(line) for line in path.with_name("metrics.jsonl").read_text().splitlines() if line.strip()]
    if ([r["epoch"] for r in rows] != list(range(1, epochs + 1))
            or any(r.get("synthetic") is not False for r in rows) or entry["epochs"] != epochs):
        raise ValueError("Incomplete or synthetic source metrics")
    if (checkpoint.get("checkpoint_type") != "swa" or checkpoint.get("selected_source") != "swa"
            or checkpoint.get("epoch") != epochs or source["selected_epoch"] != epochs
            or checkpoint.get("trained_epochs_total") != source["trained_epochs_total"]):
        raise ValueError("SWA artifact type, final epoch or lineage mismatch")
    if (checkpoint["config"] != run_config or canonical["config"] != run_config
            or resolve_config(json.loads(Path(entry["config"]).read_text())) != resolve_config(run_config)
            or checkpoint["config"]["model"] != dict(model.config)):
        raise ValueError("Source architecture or configuration mismatch")
    if (metadata.get("synthetic") is not False or (metadata.get("train_size"), metadata.get("val_size")) != (45000, 5000)
            or checkpoint["metadata"] != run_meta or canonical["metadata"] != run_meta):
        raise ValueError("Require actual source metadata and real held-out data")
    for key in ("fingerprint", "split_fingerprint", "class_names", "synthetic", "train_size", "val_size"):
        if metadata.get(key) != run_meta.get(key):
            raise ValueError("Warm-start data/split mismatch: " + key)
    parameters = count_parameters(model)
    if (not 0 < parameters <= 5_000_000 or any(v != parameters for v in
            (checkpoint.get("parameters"), canonical.get("parameters"), entry.get("parameters"), source["parameters"]))):
        raise ValueError("Actual parameter count violates source registration")
    best = json.loads(path.with_name("best.json").read_text(encoding="utf-8"))
    if (canonical.get("checkpoint_type") != "best" or best.get("epoch") != canonical.get("epoch")
            or best.get("source") != canonical.get("selected_source") or best.get("parameters") != parameters
            or best.get("accuracy") != entry.get("accuracy") or best.get("synthetic") is not False):
        raise ValueError("Canonical best registration mismatch")
    metric_path = path.with_name("swa.json")
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    score = metric.get("accuracy")
    if (metric.get("synthetic") is not False or isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError("Invalid SWA-specific validation metric")
    ancestry = verify_ancestry(checkpoint, metadata)
    verify_ancestry(canonical, metadata)
    if ancestry != source["ancestry_dependency_sha256"]:
        raise ValueError("Registered ancestry dependency set differs")
    verify_state(checkpoint["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    if file_sha256(registration_path) != registration_sha256 or file_sha256(path) != source["checkpoint_sha256"]:
        raise ValueError("Source or registration changed during admission")
    return {"checkpoint": str(path), "sha256": source["checkpoint_sha256"], "epoch": checkpoint["epoch"],
        "trained_epochs_total": checkpoint["trained_epochs_total"], "selected_source": "swa",
        "artifact_type": "swa", "validation_accuracy": float(score), "parameters": parameters,
        "source_provenance": copy.deepcopy(checkpoint["provenance"]),
        "artifact_registration": {"path": str(registration_path), "sha256": registration_sha256},
        "source_metric": {"path": str(metric_path), "sha256": file_sha256(metric_path), "field": "accuracy",
                          "accuracy": float(score), "precision": run_config["training"]["amp"]},
        "note": "Registered original project SWA weights; optimizer, EMA, SWA accumulator, RNG and schedule restart explicitly."}
