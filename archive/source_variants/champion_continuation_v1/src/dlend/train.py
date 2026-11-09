"""Training CLI: from scratch, auditable validation, single-model checkpoints."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import build_datasets
from .models import build_model, count_parameters
from .optim import (
    ModelEMA,
    classification_loss,
    freeze_bn_statistics,
    learning_rate,
    make_optimizer,
    mix_batch,
    sam_perturb,
    sam_restore,
)
from .predict import predict_probabilities
from .utils import (
    apply_overrides,
    atomic_save,
    environment,
    file_sha256,
    get_device,
    load_checkpoint,
    restore_rng,
    rng_state,
    seed_everything,
    seed_worker,
    write_json,
)

DEFAULT_TRAINING = {
    "epochs": 200, "batch_size": 128, "workers": 0, "lr": 0.1,
    "min_lr": 0.0001, "warmup_epochs": 5, "scheduler": "cosine", "optimizer": "sgd", "momentum": 0.9,
    "weight_decay": 0.0005, "nesterov": True, "label_smoothing": 0.05,
    "mixup_alpha": 0.2, "cutmix_alpha": 1.0, "mix_probability": 0.8,
    "ema_decay": 0.999, "grad_clip": 0.0, "amp": "bf16", "channels_last": True,
    "compile": False, "sam_rho": 0.0, "teacher_checkpoint": None,
    "distill_alpha": 0.5, "distill_temperature": 4.0,
    "early_stop_patience": 0, "deterministic": False, "device": "auto", "threads": 4,
    "swa_start": 0, "swa_bn_batches": 0,
}


def resolve_config(config):
    cfg = copy.deepcopy(config)
    if set(cfg) - {"seed", "model", "dataset", "training"}:
        raise ValueError("Unknown top-level config key")
    cfg.setdefault("seed", 42)
    given = cfg.setdefault("training", {})
    unknown = set(given) - set(DEFAULT_TRAINING)
    if unknown:
        raise ValueError(f"Unknown training options: {sorted(unknown)}")
    cfg["training"] = t = DEFAULT_TRAINING | given
    cfg.setdefault("model", {"depth": 28, "widen_factor": 3.5, "drop_path": 0.0, "num_classes": 10})
    cfg.setdefault("dataset", {}).setdefault("seed", cfg["seed"])
    if int(cfg["model"].get("num_classes", 10)) != 10:
        raise ValueError("This competition pipeline requires exactly 10 classes")
    for key in ("epochs", "batch_size", "threads"):
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("workers", "warmup_epochs", "early_stop_patience", "swa_start", "swa_bn_batches"):
        if not isinstance(t[key], int) or t[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("lr", "min_lr", "weight_decay", "mixup_alpha", "cutmix_alpha", "sam_rho", "grad_clip"):
        if not math.isfinite(t[key]) or t[key] < 0:
            raise ValueError(f"Invalid {key}")
    for key in ("label_smoothing", "mix_probability", "ema_decay", "distill_alpha"):
        if not 0 <= t[key] < 1 and not (key in {"mix_probability", "distill_alpha"} and t[key] == 1):
            raise ValueError(f"Invalid {key}")
    if t["lr"] <= 0 or t["min_lr"] > t["lr"] or t["warmup_epochs"] > t["epochs"]:
        raise ValueError("Require 0 <= min_lr <= lr, lr > 0, warmup_epochs <= epochs")
    if t["amp"] not in {"none", "bf16", "fp16"}:
        raise ValueError("amp must be none, bf16, or fp16")
    if t["optimizer"] not in {"sgd", "adam", "adamw", "rmsprop"}:
        raise ValueError("optimizer must be sgd, adam, adamw, or rmsprop")
    if t["scheduler"] not in {"cosine", "linear", "multistep"}:
        raise ValueError("scheduler must be cosine, linear, or multistep")
    if t["sam_rho"] and t["amp"] == "fp16":
        raise ValueError("SAM supports bf16 or full precision; use amp=bf16/none")
    if not math.isfinite(t["distill_temperature"]) or t["distill_temperature"] <= 0:
        raise ValueError("distill_temperature must be positive")
    if not math.isfinite(t["momentum"]) or not 0 <= t["momentum"] < 1:
        raise ValueError("momentum must lie in [0,1)")
    if t["optimizer"] == "sgd" and t["nesterov"] and t["momentum"] <= 0:
        raise ValueError("Nesterov SGD requires positive momentum")
    if t["swa_start"] > t["epochs"]:
        raise ValueError("swa_start must be 0 (off) or <= epochs")
    if t["swa_start"] and t["early_stop_patience"]:
        raise ValueError("SWA uses a fixed schedule; disable early stopping")
    return cfg


def autocast_context(device, amp):
    enabled = device.type == "cuda" and amp != "none"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16,
                          enabled=enabled)


@torch.inference_mode()
def evaluate_loader(model, loader, device, amp="none", channels_last=False, tta="none"):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    predictions, targets = [], []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with autocast_context(device, amp):
            probs = predict_probabilities(model, images, tta)
            if not torch.isfinite(probs).all():
                raise FloatingPointError("Non-finite validation probabilities")
            loss = F.nll_loss(probs.clamp_min(1e-12).log(), labels, reduction="sum")
        pred = probs.argmax(1)
        correct += int((pred == labels).sum())
        total += len(labels)
        loss_sum += float(loss)
        predictions.extend(pred.cpu().tolist())
        targets.extend(labels.cpu().tolist())
    if total == 0:
        raise ValueError("Empty evaluation dataset")
    return {"loss": loss_sum / total, "accuracy": correct / total,
            "count": total, "predictions": predictions, "targets": targets}


def _has_random_ancestry(provenance):
    """A continuation retains the explicit chain back to project random init."""
    if not isinstance(provenance, dict) or provenance.get("external_data") is True:
        return False
    if provenance.get("initialization") == "random":
        return True
    source = provenance.get("warm_start", {})
    return (provenance.get("initialization") == "warm_start"
            and isinstance(source, dict) and len(source.get("sha256", "")) == 64
            and _has_random_ancestry(source.get("source_provenance")))


def _validate_teacher(checkpoint, metadata):
    if not _has_random_ancestry(checkpoint.get("provenance", {})):
        raise ValueError("Teacher must be a from-scratch checkpoint produced by this project")
    teacher_meta = checkpoint["metadata"]
    for key in ("fingerprint", "split_fingerprint", "class_names", "synthetic"):
        if key not in metadata or metadata.get(key) != teacher_meta.get(key):
            raise ValueError(f"Teacher data/split mismatch ({key}); validation leakage is forbidden")


def _load_warm_start(path, model, metadata, *, artifact_registration=None, artifact_registration_sha256=None):
    """Load an auditable project best; optimizer and scheduler are never copied."""
    if artifact_registration is not None or artifact_registration_sha256 is not None:
        if not artifact_registration or not artifact_registration_sha256:
            raise ValueError("Artifact registration path and SHA must be supplied together")
        from .registered_swa_warm_start import load_registered_swa
        return load_registered_swa(path, model, metadata, artifact_registration, artifact_registration_sha256)
    path = Path(path).resolve()
    if path.name != "best.pt":
        raise ValueError("Warm start requires the original project best.pt and its best.json")
    digest = file_sha256(path)
    checkpoint = load_checkpoint(path)
    if file_sha256(path) != digest:
        raise ValueError("Warm-start checkpoint changed while it was being loaded")
    if checkpoint.get("checkpoint_type") != "best":
        raise ValueError("Warm start requires a selected best checkpoint")
    source_meta = checkpoint.get("metadata", {})
    if metadata.get("synthetic") is not False or source_meta.get("synthetic") is not False:
        raise ValueError("Warm start requires real data; synthetic checkpoints are not eligible")
    for key in ("fingerprint", "split_fingerprint", "class_names"):
        if key not in metadata or source_meta.get(key) != metadata[key]:
            raise ValueError(f"Warm-start data/split mismatch: {key}")
    if not _has_random_ancestry(checkpoint.get("provenance")):
        raise ValueError("Warm start requires a from-scratch project checkpoint")
    if checkpoint.get("config", {}).get("model") != dict(model.config):
        raise ValueError("Warm-start model architecture must match exactly")
    info = json.loads(path.with_name("best.json").read_text(encoding="utf-8-sig"))
    epoch, source = checkpoint.get("epoch"), checkpoint.get("selected_source")
    trained_epochs = checkpoint.get("trained_epochs_total", epoch)
    if (not isinstance(epoch, int) or epoch < 0 or not isinstance(trained_epochs, int)
            or trained_epochs < 1 or source not in {"model", "ema", "swa"}):
        raise ValueError("Warm start requires a trained, selected checkpoint")
    if (info.get("epoch") != epoch or info.get("source") != source
            or info.get("parameters") != count_parameters(model)
            or info.get("synthetic") is not False or info.get("selection_metric") != "validation_accuracy"):
        raise ValueError("Warm-start best.json does not match the checkpoint")
    score = info.get("accuracy")
    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Warm-start source validation accuracy is invalid")
    state = checkpoint["model"]
    if not any(int(v) > 0 for key, v in state.items() if key.endswith("num_batches_tracked")):
        raise ValueError("Warm-start checkpoint has no evidence of training (batch statistics)")
    if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError("Warm-start weights contain non-finite values")
    model.load_state_dict(state, strict=True)
    return {"checkpoint": str(path), "sha256": digest, "epoch": epoch,
            "trained_epochs_total": trained_epochs, "selected_source": source,
            "validation_accuracy": score, "parameters": count_parameters(model),
            "source_provenance": copy.deepcopy(checkpoint["provenance"]),
            "note": "Selected project weights; optimizer, EMA, RNG and LR schedule restart explicitly."}


def train(config: dict, output: Path, resume: str | None = None, max_epochs: int | None = None,
          stop_file: Path | None = None, init_checkpoint: str | None = None,
          init_artifact_registration: str | None = None, init_artifact_registration_sha256: str | None = None):
    if init_artifact_registration is not None or init_artifact_registration_sha256 is not None:
        if not init_checkpoint or resume or not init_artifact_registration or not init_artifact_registration_sha256:
            raise ValueError("Artifact registration pair requires a fresh --init-checkpoint, never --resume")
    cfg = resolve_config(config)
    t = cfg["training"]
    output = Path(output)
    if resume and init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if init_checkpoint:
        if Path(init_checkpoint).resolve().parent == output.resolve():
            raise ValueError("Warm start must use a new output directory, never its source directory")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Warm start requires a new empty output directory")
    if resume and Path(resume).resolve().parent != output.resolve():
        raise ValueError("Resume must use its original output directory to preserve best.pt and metrics")
    if not resume and output.exists() and any(output.glob("*.pt")):
        raise FileExistsError("Run directory has checkpoints. Choose a new output or --resume last.pt")
    if max_epochs is not None and max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(t["threads"])
    seed_everything(cfg["seed"], t["deterministic"])
    device = get_device(t["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda" and t["amp"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU lacks bfloat16 support; select amp=fp16 or none")
    train_set, val_set, metadata = build_datasets(cfg, output)
    if len(metadata["class_names"]) != 10:
        raise ValueError("Require all ten classes in the supplied training data")
    if val_set is None and t["early_stop_patience"]:
        raise ValueError("Early stopping requires a validation set")
    model = build_model(cfg["model"]).to(device)
    cfg["model"] = dict(model.config)
    parameters = count_parameters(model)
    if t["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    warm_start = _load_warm_start(init_checkpoint, model, metadata,
                                  artifact_registration=init_artifact_registration,
                                  artifact_registration_sha256=init_artifact_registration_sha256) if init_checkpoint else None
    if warm_start and val_set is None:
        raise ValueError("Warm start requires the original held-out validation split")
    optimizer = make_optimizer(model, t)
    ema = ModelEMA(model, t["ema_decay"]) if t["ema_decay"] else None
    swa = torch.optim.swa_utils.AveragedModel(model, device=device) if t["swa_start"] else None
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and t["amp"] == "fp16")
    loader_generator = torch.Generator().manual_seed(cfg["seed"])
    loader_kwargs = {"batch_size": t["batch_size"], "num_workers": t["workers"],
                     "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker,
                     "persistent_workers": False}
    train_loader = DataLoader(train_set, shuffle=True, generator=loader_generator, **loader_kwargs)
    # Validation uses inexpensive, deterministic transforms on cached 32x32 data.
    # Avoid spawning fresh Windows workers twice each epoch for raw and EMA eval.
    validation_kwargs = dict(loader_kwargs, num_workers=0)
    val_loader = DataLoader(val_set, shuffle=False, **validation_kwargs) if val_set is not None else None
    teacher = None
    teacher_sha256 = None
    if t["teacher_checkpoint"]:
        teacher_sha256 = file_sha256(t["teacher_checkpoint"])
        teacher_ckpt = load_checkpoint(t["teacher_checkpoint"])
        _validate_teacher(teacher_ckpt, metadata)
        teacher = build_model(teacher_ckpt["config"]["model"]).to(device).eval().requires_grad_(False)
        teacher.load_state_dict(teacher_ckpt["model"])
    start_epoch, best_accuracy, best_epoch = 0, -1.0, 0
    if resume:
        checkpoint = load_checkpoint(resume)
        if checkpoint.get("checkpoint_type") != "last":
            raise ValueError("Only last.pt contains resumable optimizer/RNG state")
        # Older checkpoints predate optional defaults such as scheduler=cosine.
        # Normalize only missing defaults; any effective configuration change
        # (including a different schedule) must still fail exact comparison.
        if resolve_config(checkpoint["config"]) != cfg:
            raise ValueError("Resume requires the identical resolved config; extend via a new explicit experiment")
        if checkpoint.get("provenance", {}).get("teacher_sha256") != teacher_sha256:
            raise ValueError("Teacher checkpoint changed since training began")
        warm_start = copy.deepcopy(checkpoint.get("provenance", {}).get("warm_start"))
        for key in ("fingerprint", "split_fingerprint", "class_names"):
            if checkpoint["metadata"].get(key) != metadata.get(key):
                raise ValueError(f"Resume dataset mismatch: {key}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        if ema:
            ema.module.load_state_dict(checkpoint["ema"])
            ema.updates = checkpoint["ema_updates"]
        if swa:
            swa.load_state_dict(checkpoint["swa"])
        start_epoch, best_accuracy, best_epoch = checkpoint["epoch"], checkpoint["best_accuracy"], checkpoint["best_epoch"]
        loader_generator.set_state(checkpoint["loader_rng"])
        restore_rng(checkpoint["rng"])
        # Discard uncheckpointed log rows after a crash; do not duplicate completed epochs.
        metrics_path = output / "metrics.jsonl"
        if metrics_path.exists():
            lines = [s for s in metrics_path.read_text(encoding="utf-8").splitlines() if s.strip()]
            rows = []
            for index, line in enumerate(lines):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if index != len(lines) - 1:
                        raise
            metrics_path.write_text("".join(json.dumps(r) + "\n" for r in rows if r["epoch"] <= start_epoch and r.get("stage") != "swa"), encoding="utf-8")
    network = torch.compile(model) if t["compile"] else model
    write_json(cfg, output / "config.json")
    write_json(metadata, output / "dataset.json")
    write_json(environment(), output / "environment.json")
    if warm_start:
        write_json(warm_start, output / "initialization.json")
    synthetic = bool(metadata.get("synthetic", False))
    print(json.dumps({"device": str(device), "parameters": parameters,
                      "train_size": len(train_set), "validation_size": len(val_set) if val_set is not None else 0,
                      "synthetic": synthetic, "status": "SMOKE ONLY" if synthetic else "training"}), flush=True)

    def inference_checkpoint(state_model, epoch, source):
        return {"checkpoint_type": "best" if val_loader else "final", "model": state_model.state_dict(), "ema": None,
                "config": cfg, "metadata": metadata, "epoch": epoch, "selected_source": source,
                "trained_epochs_total": (warm_start["trained_epochs_total"] if warm_start else 0) + epoch,
                "parameters": parameters, "provenance": {"initialization": "warm_start" if warm_start else "random",
                "warm_start": warm_start, "external_data": False,
                "teacher_checkpoint": t["teacher_checkpoint"], "teacher_sha256": teacher_sha256,
                "note": "User-supplied data; no external data downloaded by code"}}

    def select(state_model, result, epoch, source):
        nonlocal best_accuracy, best_epoch
        if result["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = result["accuracy"], epoch
            best = {"epoch": epoch, "accuracy": best_accuracy, "source": source,
                    "parameters": parameters, "synthetic": synthetic, "selection_metric": "validation_accuracy"}
            atomic_save(inference_checkpoint(state_model, epoch, source), output / "best.pt")
            write_json(best, output / "best.json")

    if init_checkpoint:
        # Keep the inherited model if every continuation epoch is worse.
        initial_result = evaluate_loader(model, val_loader, device, t["amp"], t["channels_last"])
        select(model, initial_result, 0, "model")
        write_json({"accuracy": initial_result["accuracy"], "loss": initial_result["loss"],
                    "count": initial_result["count"], "split": "validation", "epoch": 0,
                    "split_fingerprint": metadata["split_fingerprint"]}, output / "initial_validation.json")

    end_epoch = min(t["epochs"], max_epochs) if max_epochs else t["epochs"]
    total_steps = t["epochs"] * len(train_loader)
    final_epoch = start_epoch
    for epoch in range(start_epoch, end_epoch):
        if stop_file is not None and Path(stop_file).exists():
            print("Stop requested; saved epoch checkpoint is preserved.", flush=True)
            break
        network.train()
        started = time.perf_counter()
        train_loss, train_correct, seen = 0.0, 0.0, 0
        for batch_index, (images, labels) in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_index
            lr = learning_rate(step, total_steps, t["warmup_epochs"] * len(train_loader),
                               t["lr"], t["min_lr"], t["scheduler"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if t["channels_last"]:
                images = images.contiguous(memory_format=torch.channels_last)
            images, soft_targets = mix_batch(images, labels, 10, t["mixup_alpha"], t["cutmix_alpha"],
                                             t["mix_probability"], t["label_smoothing"])
            teacher_logits = None
            if teacher is not None:
                with torch.no_grad(), autocast_context(device, t["amp"]):
                    teacher_logits = teacher(images)
            optimizer.zero_grad(set_to_none=True)
            stochastic_state = rng_state() if t["sam_rho"] else None
            with autocast_context(device, t["amp"]):
                logits = network(images)
                loss = classification_loss(logits, soft_targets, teacher_logits, t["distill_alpha"], t["distill_temperature"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch+1}, batch {batch_index}")
            scaler.scale(loss).backward()
            if t["sam_rho"]:
                saved = sam_perturb(model, t["sam_rho"])
                optimizer.zero_grad(set_to_none=True)
                try:
                    restore_rng(stochastic_state)
                    with freeze_bn_statistics(model), autocast_context(device, t["amp"]):
                        second_loss = classification_loss(network(images), soft_targets, teacher_logits,
                                                          t["distill_alpha"], t["distill_temperature"])
                    if not torch.isfinite(second_loss):
                        raise FloatingPointError("Non-finite SAM second loss")
                    second_loss.backward()
                finally:
                    sam_restore(saved)
            scaler.unscale_(optimizer)
            if not scaler.is_enabled():
                gradients_finite = torch.stack([p.grad.isfinite().all() for p in model.parameters() if p.grad is not None]).all()
                if not gradients_finite:
                    raise FloatingPointError("Non-finite gradients; optimizer step cancelled")
            if t["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"], error_if_nonfinite=True)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if ema and scaler.get_scale() >= old_scale:
                ema.update(model)
            n = len(labels)
            train_loss += float(loss.detach()) * n
            # Diagnostic soft-target agreement, NOT a hard-label training/test accuracy.
            train_correct += float(soft_targets.gather(1, logits.argmax(1, keepdim=True)).sum().detach())
            seen += n
        if swa and epoch + 1 >= t["swa_start"]:
            swa.update_parameters(model)
        elapsed_train = time.perf_counter() - started
        val = evaluate_loader(model, val_loader, device, t["amp"], t["channels_last"]) if val_loader else None
        ema_val = evaluate_loader(ema.module, val_loader, device, t["amp"], t["channels_last"]) if val_loader and ema else None
        if val:
            select(model, val, epoch + 1, "model")
        if ema_val:
            select(ema.module, ema_val, epoch + 1, "ema")
        selected = max([(v["accuracy"], name) for v, name in [(val, "model"), (ema_val, "ema")] if v], default=(None, None))
        metrics = {"epoch": epoch + 1, "train_loss": train_loss / seen, "train_acc": train_correct / seen,
                   "train_acc_definition": "soft_target_agreement", "val_loss": val["loss"] if val else None,
                   "val_acc": val["accuracy"] if val else None, "ema_val_loss": ema_val["loss"] if ema_val else None,
                   "ema_val_acc": ema_val["accuracy"] if ema_val else None, "selection_acc": selected[0],
                   "selection_source": selected[1], "lr": lr, "seconds": time.perf_counter() - started,
                   "images_per_second": seen / elapsed_train, "synthetic": synthetic}
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        last = inference_checkpoint(model, epoch + 1, "model")
        last.update(checkpoint_type="last", optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                    ema=ema.module.state_dict() if ema else None, ema_updates=ema.updates if ema else 0,
                    swa=swa.state_dict() if swa else None, best_accuracy=best_accuracy, best_epoch=best_epoch,
                    rng=rng_state(), loader_rng=loader_generator.get_state())
        atomic_save(last, output / "last.pt")
        final_epoch = epoch + 1
        print(json.dumps(metrics, allow_nan=False), flush=True)
        if t["early_stop_patience"] and epoch + 1 - best_epoch >= t["early_stop_patience"]:
            break
    # SWA is finalized only after the configured schedule, never at a temporary max-epochs pause.
    if swa and final_epoch == t["epochs"] and int(swa.n_averaged) > 0:
        bn_loader = train_loader
        if t["swa_bn_batches"]:
            import itertools
            bn_loader = itertools.islice(train_loader, t["swa_bn_batches"])
        torch.optim.swa_utils.update_bn(bn_loader, swa, device=device)
        swa.module.eval()
        swa_checkpoint = inference_checkpoint(swa.module, final_epoch, "swa")
        swa_checkpoint["checkpoint_type"] = "swa"
        atomic_save(swa_checkpoint, output / "swa.pt")
        if val_loader:
            swa_val = evaluate_loader(swa.module, val_loader, device, t["amp"], t["channels_last"])
            select(swa.module, swa_val, final_epoch, "swa")
            write_json({"accuracy": swa_val["accuracy"], "loss": swa_val["loss"], "synthetic": synthetic}, output / "swa.json")
    if final_epoch == t["epochs"]:
        # Refit policy must be chosen before seeing test labels; default final uses EMA when enabled.
        final_model = ema.module if ema else model
        final_checkpoint = inference_checkpoint(final_model, final_epoch, "ema" if ema else "model")
        final_checkpoint["checkpoint_type"] = "final"
        atomic_save(final_checkpoint, output / "final.pt")
    print(json.dumps({"completed_epochs": final_epoch, "configured_epochs": t["epochs"],
                      "best_validation_accuracy": best_accuracy if val_loader else None,
                      "parameters": parameters, "synthetic": synthetic}), flush=True)
    return {"parameters": parameters, "best_accuracy": best_accuracy if val_loader else None,
            "completed_epochs": final_epoch, "synthetic": synthetic}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume")
    initialization.add_argument("--init-checkpoint", help="Continue a project best.pt in a new empty run with a fresh schedule")
    parser.add_argument("--init-artifact-registration", help="Frozen explicit original SWA registration; requires expected SHA")
    parser.add_argument("--init-artifact-registration-sha256")
    parser.add_argument("--stop-file", type=Path, help="Stop at an epoch boundary if this file exists")
    parser.add_argument("--max-epochs", type=int, help="Pause after this absolute epoch; retain original LR schedule")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    train(apply_overrides(cfg, args.set), Path(args.output), args.resume, args.max_epochs,
          args.stop_file, args.init_checkpoint, args.init_artifact_registration, args.init_artifact_registration_sha256)


if __name__ == "__main__":
    main()
