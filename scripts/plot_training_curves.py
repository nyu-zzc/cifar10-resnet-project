"""Plot complete, unsmoothed historical logs; no training or data inference."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = [('base400', 'A1: from scratch (400)'), ('mild60', 'A2: mild appearance (60)'),
        ('robust300', 'B1: SE from scratch (300)'), ('robust60', 'B2: robust SWA (60)'),
        ('robust_refine50', 'C: SWA continuation (50)')]


def render():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    out = ROOT / 'results/figures'
    out.mkdir(parents=True, exist_ok=True)
    for kind in ['validation', 'loss']:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
        for axis, (name, title) in zip(axes.flat, RUNS):
            config = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
            rows = [json.loads(line) for line in (ROOT / 'results/runs' / name / 'metrics.jsonl').read_text().splitlines() if line]
            epochs = [r['epoch'] for r in rows]
            assert epochs == list(range(1, config['training']['epochs'] + 1))
            assert all(r['synthetic'] is False for r in rows)
            if kind == 'validation':
                axis.plot(epochs, [100 * r['val_acc'] for r in rows], color='#3267a7', lw=1.15, label='Raw, single view')
                axis.plot(epochs, [100 * r['ema_val_acc'] for r in rows], color='#c06b25', lw=1.15, label='EMA, single view')
                axis.set_ylabel('Validation accuracy (%)')
                axis.set_ylim(0 if name in {'base400', 'robust300'} else 94, 100)
            else:
                axis.plot(epochs, [r['train_loss'] for r in rows], color='#3267a7', lw=1.15, label='Mixed-target train loss')
                axis.plot(epochs, [r['val_loss'] for r in rows], color='#c06b25', lw=1.15, label='Raw validation NLL')
                axis.set_ylabel('Logged loss (different objectives)')
                axis.set_ylim(bottom=0)
            axis.set(title=title, xlabel='Epoch within this phase', xlim=(1, len(rows)))
            axis.grid(alpha=.18)
            axis.legend(fontsize=8, loc='lower right' if kind == 'validation' else 'upper right')
        note = axes.flat[-1]
        note.axis('off')
        note.text(.02, .95,
                  'Real historical logs; all 870 phase epochs shown.\n\n'
                  'Each panel is a different training phase.\n'
                  'Validation uses the same 5,000 held-out images.\n'
                  'Training-time validation: CUDA BF16, no TTA.\n\n'
                  'SWA after BN recalibration and CPU FP32 + flip\n'
                  'are separate endpoints, not these line values.\n\n'
                  'No smoothing, error bars or invented repeats.\n'
                  'Mixed-label train loss is not directly comparable\n'
                  'to clean validation NLL.', va='top', fontsize=11, linespacing=1.5)
        fig.suptitle('Recorded training history: ' + ('validation' if kind == 'validation' else 'loss'), fontsize=16)
        fig.savefig(out / f'{kind}_curves.png', dpi=160)
        plt.close(fig)
    return [out / 'validation_curves.png', out / 'loss_curves.png']


if __name__ == '__main__':
    for path in render():
        print(path.relative_to(ROOT).as_posix())
