"""CPU-only package checks; synthetic inputs are not accuracy measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch

from dlend.models import build_model, count_parameters
from dlend.train import resolve_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/package_verification.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    manifest = json.loads((ROOT / 'provenance/source_mapping.json').read_text())
    for item in manifest['exact_copies']:
        actual = hashlib.sha256((ROOT / item['published_path']).read_bytes()).hexdigest()
        assert actual == item['original_sha256'] == item['published_sha256'], item['published_path']
    models = []
    for path in sorted((ROOT / 'configs').glob('*.json')):
        cfg = resolve_config(json.loads(path.read_text()))
        model = build_model(cfg['model']).eval()
        with torch.inference_mode():
            logits = model(torch.zeros(2, 3, 32, 32))
        assert list(logits.shape) == [2, 10] and torch.isfinite(logits).all()
        expected = 4_480_474 if cfg['model']['widen_factor'] == 3.5 else 4_948_626
        assert count_parameters(model) == expected
        models.append({'config': path.name, 'parameters': expected, 'stage_blocks': list(model.stage_blocks),
                       'stage_widths': list(model.stage_widths), 'kernel_sizes': list(model.kernel_sizes),
                       'output_shape': list(logits.shape), 'finite': True})
        logs = ROOT / 'results/runs' / path.stem / 'metrics.jsonl'
        rows = [json.loads(line) for line in logs.read_text().splitlines() if line]
        assert [r['epoch'] for r in rows] == list(range(1, cfg['training']['epochs'] + 1))
        assert all(r['synthetic'] is False for r in rows)
    result = {'status': 'passed', 'device': 'cpu', 'python': sys.version.split()[0], 'torch': torch.__version__,
              'exact_copy_hash_checks': len(manifest['exact_copies']), 'models': models,
              'scope': 'Source hashes, configs, parameter counts, synthetic finite forward, real-log completeness. No training, accuracy evaluation or test inference.'}
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
