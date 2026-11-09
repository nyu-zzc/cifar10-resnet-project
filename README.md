# CIFAR-10 residual networks under 5M parameters

[Project repository](https://github.com/nyu-zzc/cifar10-resnet-project) · [中文说明](README_zh.md) · [Executed notebook](notebooks/reproduction.ipynb) · [Complete experiment archive](archive/README.md)

Code and recorded results for the [CIFAR-10 ResNet Architecture Challenge](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl). Models start from random initialization; continuation and distillation use only weights trained within this project. No external pretrained weights or additional training dataset were used.

The repository contains the original training/prediction source, five directly checked reproduction configurations, an executed notebook, and all **79 historical tasks / 7,810 phase epochs** as complete portable configurations, aggregate logs and matching source snapshots. Competition images, per-image labels/predictions, account information and checkpoints are **not distributed**. Obtain the competition archive through its official access process. Reading the notebook does not start training.

## Results and their distinct identities

| Role | Model / selected artifact | Parameters | Validation FP32 + flip | Public | Private |
|---|---|---:|---:|---:|---:|
| Best public score; checked in screenshot | WRN-28-3.5, mild60 SWA | 4,480,474 | 97.36% | 89.4% | 86.1% |
| Selected final result | WRN-28-3.625 + SE, robust60 SWA | 4,948,626 | 97.40% | 88.6% | **86.3%** |
| Highest visible private score; unchecked | Same SE architecture, robust_refine50 EMA 29 | 4,948,626 | 97.48% | 88.6% | 87.3% |

Official scores were read from the user's submission-page screenshot on 2026-11-09 and matched by filename to local CSV/checkpoint records. Uploaded bytes were not independently retrieved. The visible 87.3% submission was **not selected**, so it is not the team's selected final score. The original screenshot contains account UI and is not redistributed; see the [sanitized official-results summary](results/official_results_public_summary.json).

Local validation reuses a fixed5,000-image split; training uses 45,000 images, and the competition has2,000 hidden test images. These are different evaluation scopes. The 79 tasks comprise39 screens,18 full scratch runs and 22 continuations. Completion does not establish global convergence, an exhaustive search, or effectiveness of every method.

## Environment and installation

Historical environment: Python 3.12.10, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, NumPy 2.5.4, CUDA12.8, Windows 11 and NVIDIA GeForce RTX5070 Ti. These are the versions actually used, not a claim about current releases. From the repository root, in a Python 3.12 environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/verify_package.py
```

On Linux/macOS activate with `source .venv/bin/activate`. For notebook/CPU checks, install matching CPU wheels from PyTorch's official index instead. Training configurations retain CUDA, BF16, batch 128 and workers 2. Record any hardware/precision/worker changes. Training used `deterministic=false`; reruns are not guaranteed to reproduce identical bytes, selected epochs or scores.

## Prepare the competition data

```powershell
python scripts/prepare_competition.py --archive data/competition.zip --output data/competition
```

The original ZIP contains `train_labels.csv` (id,filename,label), `test.csv` (id,filename), `sample_submission.csv` (id,label), and 32×32 RGB images. Preparation creates `train.npz`, `test_images/`, the three CSVs and an audit. It decodes training images directly from ZIP, without requiring50,000 extracted files. Placeholder submission labels are never used for training/evaluation. No dataset is downloaded by the code.

Indices0–9 map to airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck. Submission labels remain strings "0"–"9". The seed 42 split groups exact duplicates on one side. Original auditing found no train/test exact-pixel overlap. Aggregate counts and fingerprints are in [data_summary.json](results/data_summary.json).

## Route A: public-best mild appearance model

```powershell
python -m dlend.train --config configs/base400.json --output runs/base400
python -m dlend.train --config configs/mild60.json --output runs/mild60 --init-checkpoint runs/base400/best.pt
```

The scratch phase runs400 epochs; the historical `best.pt` was raw epoch 396. The 60-epoch continuation restarts the optimizer, EMA, RNG and schedule, with lr 0.003→0.00001 and warmup 3. Average raw weights from epochs45–60 and recalibrate BN on training data. The reported artifact is **runs/mild60/swa.pt**, not its `best.pt`. Compute is400+60=460 epochs; selected-weight lineage is396+60.

Architecture:3×3 stem3→16; three stages of4 preactivation basic blocks each; two3×3 convolutions per block; channels 56/112/224; stride 2 at the first block of stages2/3;1×1 projection shortcuts; final BN-ReLU, global average pooling and 10-class linear head. No SE, DropPath or dropout. “28” is the WRN depth convention, not28 residual blocks.

Training uses reflection-padded random crop, horizontal flip, RandAugment N2/M7, SGD momentum 0.9/Nesterov, weight decay 0.0005 and EMA 0.999. Mixup alpha 0.2 and CutMix alpha 1 are alternatives within total mixing probability 0.8; label smoothing0.05. Mild60 adds ColorJitter0.15, grayscale probability 0.05 and Gaussian blur probability 0.03 (kernel 3, sigma 0.1–1.0).

## Route B: selected-final robust SWA model

```powershell
python -m dlend.train --config configs/robust300.json --output runs/robust300
python -m dlend.train --config configs/robust60.json --output runs/robust60 --init-checkpoint runs/robust300/best.pt
```

Width3.625 gives channels 58/116/232; SE ratio 0.25 gives4,948,626 total parameters. The 300-epoch scratch run selected EMA 285, followed by60 epochs and **runs/robust60/swa.pt**. Compute:360 epochs; selected lineage:285+60=345.

This branch uses RandAugment M5, ColorJitter0.2, grayscale0.1, blur0.05, **Mixup alpha 0.4, CutMix alpha 1, mixing probability 0.6 and label smoothing0**. Other exact values are in the configurations. Because architecture and augmentation change together, differences between routes are not isolated single-factor effects.

## Route C: observed private maximum, not selected

After route B, continue from its original SWA artifact:

```powershell
python scripts/train_from_swa.py --config configs/robust_refine50.json --output runs/robust_refine50 --init-checkpoint runs/robust60/swa.pt --check-only
python scripts/train_from_swa.py --config configs/robust_refine50.json --output runs/robust_refine50 --init-checkpoint runs/robust60/swa.pt
```

This 50-epoch phase uses lr 0.0005→0.00001, no warmup and SWA start 38. The actual submitted artifact was **runs/robust_refine50/best.pt**, containing EMA 29, not SWA. Compute:300+60+50=410 epochs; selected lineage:285+60+29=374.

The new portable admission wrapper checks the original `swa.pt`, its own metric, complete logs, dataset/split fingerprints, architecture, finite weights and declared project random ancestry, then calls the unchanged training loop. It restarts optimizer/EMA/SWA accumulator/RNG/schedule. Use `--expected-source-sha256 HEX` to pin a particular artifact. Historical admission additionally depended on private machine-local signed authorization files, which are not redistributed. This admission adaptation does not change training mathematics; trust only your own or otherwise trusted project checkpoints, since metadata alone cannot authenticate arbitrary third-party claims.

## Evaluate and predict in CPU FP32 with two threads

```powershell
python scripts/inference_cpu.py evaluate --threads 2 --checkpoint runs/mild60/swa.pt --data-root data/competition --tta flip --batch-size 64 --output runs/mild60/validation_flip.json
python scripts/inference_cpu.py predict --threads 2 --checkpoint runs/mild60/swa.pt --test-csv data/competition/test.csv --image-dir data/competition/test_images --path-column filename --sample-submission data/competition/sample_submission.csv --output submissions/mild60.csv --tta flip --batch-size 64
```

Use the corresponding artifact paths for routes B/C. The wrapper explicitly fixes CPU and threads 2; the original evaluator otherwise uses the checkpoint's training thread count, so environment variables alone are insufficient. CPU disables CUDA autocast, yielding FP32. Model weights, data transforms, split verification and probability averaging are unchanged. Training-time BF16 single-view scores are not interchangeable with these endpoints.

Both `best.pt` and `swa.pt` contain the selected inference weights in `model`; retain default `--weights model`. Flip averages original/flip probabilities from one model, without increasing parameter count. Output follows the sample submission's exact ID order and is never uploaded automatically. Checkpoints are not bundled, so train the chosen route or obtain trusted original weights before inference.

Resume a stopped run with its identical config and original output directory, for example:

```powershell
python -m dlend.train --config configs/mild60.json --output runs/mild60 --resume runs/mild60/last.pt
```

The SWA-started phase also resumes through `dlend.train --resume`; do not restart it with `--init-checkpoint` into an existing directory.

## Notebook, archive and verification boundaries

The [executed notebook](notebooks/reproduction.ipynb) reads all 870 actual epochs from the five core phases, plots unsmoothed curves and checks parameter counts/synthetic CPU forwards. It does not train or infer on competition images. Saved outputs use the explicit sequential IPython harness, recorded in notebook metadata; Jupyter Run All is also supported.

```powershell
python scripts/execute_notebook.py
python scripts/verify_package.py
python -m dlend.train --help
python -m dlend.predict --help
```

With Mixup/CutMix, `train_acc` means soft-target agreement, not clean training accuracy. Training loss and validation NLL use different objectives. SWA retains the cosine tail, so it is an SWA-inspired variant rather than the original paper's constant/cyclic-LR recipe. Repeated model selection on the same validation split creates selection bias; it does not prove private-test improvement.

[Source mapping](provenance/source_mapping.json) records original/published hashes. The five main configs change only `dataset.root` to `data/competition`. The [archive](archive/README.md) includes all 79 full portable configs,7,810 epochs of aggregate logs, initialization dependencies and five matching source snapshots. The Efficient ResNets vendor's original MIT notice is preserved. Private historical authorization records are omitted; not all 79 training commands were rerun in this package. The checked core routes and complete historical evidence must not be confused with a guarantee of byte-identical reproduction without checkpoints.

See [LICENSE](LICENSE), [REFERENCES.md](REFERENCES.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The project MIT license does not replace dataset or third-party software licenses.
