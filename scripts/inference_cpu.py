"""Run the original evaluate/predict CLI in FP32 on CPU with explicit threads.

The original evaluator uses the training checkpoint's thread setting. This
wrapper changes only that runtime setting in memory, leaving checkpoint bytes,
model weights, transforms, split checks and prediction mathematics unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('mode', choices=['evaluate', 'predict'])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    args, remaining = parser.parse_known_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    import torch

    torch.set_num_threads(args.threads)
    if args.mode == 'evaluate':
        from dlend import evaluate

        original_loader = evaluate.load_checkpoint

        def load_with_threads(*a, **kw):
            checkpoint = original_loader(*a, **kw)
            checkpoint['config']['training']['threads'] = args.threads
            return checkpoint

        evaluate.load_checkpoint = load_with_threads
        evaluate.main(remaining + ['--device', 'cpu'])
    else:
        from dlend import predict

        predict.main(remaining + ['--device', 'cpu'])


if __name__ == '__main__':
    main()
