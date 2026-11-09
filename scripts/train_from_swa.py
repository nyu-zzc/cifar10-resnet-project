"""Portable admission of project SWA weights, followed by the unchanged trainer.

The historical continuation bound admission to machine-local authorization files.
This wrapper replaces only that admission layer; it never renames SWA as best.
No optimizer, RNG, EMA or SWA accumulator is copied into the new training phase.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from dlend import train as trainer
from dlend.models import build_model, count_parameters
from dlend.utils import apply_overrides, file_sha256, load_checkpoint


def admit_swa(path, model, metadata, expected_sha256=None):
    path = Path(path).resolve()
    before = file_sha256(path)
    if expected_sha256 and before.lower() != expected_sha256.lower():
        raise ValueError('Source SHA256 differs from the explicitly requested artifact')
    if path.name != 'swa.pt':
        raise ValueError('Use original swa.pt, not a renamed alias')
    checkpoint = load_checkpoint(path, device='cpu')
    cfg = json.loads(path.with_name('config.json').read_text(encoding='utf-8-sig'))
    saved_metadata = json.loads(path.with_name('dataset.json').read_text(encoding='utf-8-sig'))
    metric = json.loads(path.with_name('swa.json').read_text(encoding='utf-8-sig'))
    rows = [json.loads(line) for line in path.with_name('metrics.jsonl').read_text().splitlines() if line.strip()]
    epochs = cfg['training']['epochs']
    if ([r['epoch'] for r in rows] != list(range(1, epochs + 1))
            or any(r.get('synthetic') is not False for r in rows)):
        raise ValueError('Require a completed real-data source run with continuous epoch logs')
    if (checkpoint.get('checkpoint_type') != 'swa' or checkpoint.get('selected_source') != 'swa'
            or checkpoint.get('epoch') != epochs or checkpoint.get('config') != cfg
            or checkpoint.get('metadata') != saved_metadata or cfg['model'] != dict(model.config)):
        raise ValueError('Source artifact type, epoch, configuration or metadata mismatch')
    if metadata.get('synthetic') is not False or not trainer._has_random_ancestry(checkpoint.get('provenance')):
        raise ValueError('Require real project data and declared random-initialized project ancestry')
    for key in ['fingerprint', 'split_fingerprint', 'class_names', 'synthetic', 'train_size', 'val_size']:
        if key not in metadata or metadata[key] != saved_metadata.get(key):
            raise ValueError('Source and continuation dataset mismatch: ' + key)
    score = metric.get('accuracy')
    if (metric.get('synthetic') is not False or isinstance(score, bool)
            or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError('Invalid SWA-specific source metric')
    parameters = count_parameters(model)
    if parameters != checkpoint.get('parameters') or parameters > 5_000_000:
        raise ValueError('Source parameter count mismatch or over budget')
    state = checkpoint['model']
    if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError('Source weights contain nonfinite values')
    counters = [v for k, v in state.items() if k.endswith('num_batches_tracked')]
    if not counters or any(int(v) <= 0 for v in counters):
        raise ValueError('Source batch statistics do not show completed calibration')
    total = checkpoint.get('trained_epochs_total', checkpoint['epoch'])
    if not isinstance(total, int) or total < epochs:
        raise ValueError('Invalid source training lineage')
    model.load_state_dict(state, strict=True)
    if file_sha256(path) != before:
        raise ValueError('Source changed during admission')
    return {'checkpoint': str(path), 'sha256': before, 'epoch': epochs,
            'trained_epochs_total': total, 'selected_source': 'swa', 'artifact_type': 'swa',
            'validation_accuracy': float(score), 'parameters': parameters,
            'source_provenance': copy.deepcopy(checkpoint['provenance']),
            'note': 'Portable project SWA admission; optimizer, EMA, SWA accumulator, RNG and schedule restart.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--init-checkpoint', required=True)
    parser.add_argument('--expected-source-sha256')
    parser.add_argument('--check-only', action='store_true', help='Read source admission only; no data loading, forward or training')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    args = parser.parse_args()
    cfg = trainer.resolve_config(apply_overrides(json.loads(Path(args.config).read_text()), args.set))
    if args.check_only:
        source = load_checkpoint(args.init_checkpoint, device='cpu')
        receipt = admit_swa(args.init_checkpoint, build_model(cfg['model']), source['metadata'], args.expected_source_sha256)
        print(json.dumps({k: v for k, v in receipt.items() if k not in {'checkpoint', 'source_provenance'}}, indent=2))
        return

    def load(path, model, metadata):
        return admit_swa(path, model, metadata, args.expected_source_sha256)

    trainer._load_warm_start = load
    trainer.train(cfg, Path(args.output), init_checkpoint=args.init_checkpoint)


if __name__ == '__main__':
    main()
