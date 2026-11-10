# Reproducing the models

This guide covers the three main submission routes: the mild-appearance WRN, the robust SWA model used for the final result, and its 50-epoch refinement. Run the commands from the repository root.

## 1. Set up the environment

I used Python 3.12.10, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, NumPy 2.5.4, Windows 11, and an NVIDIA GeForce RTX 5070 Ti.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/verify_package.py
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`. For notebook and CPU checks, install the matching CPU builds of PyTorch and torchvision. The training configurations use CUDA, BF16, batch size 128, and two data-loader workers.

The original training used nondeterministic CUDA execution, so checkpoint selection and scores can vary slightly between runs.

## 2. Prepare the competition data

Download the image archive from the [competition page](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl) and save it as `data/competition.zip`.

```powershell
python scripts/prepare_competition.py --archive data/competition.zip --output data/competition
```

The archive contains `train_labels.csv` (`id,filename,label`), `test.csv` (`id,filename`), `sample_submission.csv` (`id,label`), and 32 × 32 RGB images. The preparation script creates `train.npz`, `test_images/`, the CSV files, and a data audit.

The class order is airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck, numbered 0–9. Seed 42 defines the fixed 45,000/5,000 split and keeps exact duplicates together. The [data summary](../results/data_summary.json) records the counts and fingerprints.

## 3. Choose a training route

Each route starts with a full training run. Continuation stages load its saved weights and start a new optimizer and learning-rate schedule.

### A. Mild appearance WRN

```powershell
python -m dlend.train --config configs/base400.json --output runs/base400
python -m dlend.train --config configs/mild60.json --output runs/mild60 --init-checkpoint runs/base400/best.pt
```

The first phase trains for 400 epochs. In my run, `best.pt` contained the raw weights from epoch 396. The next phase adds 60 epochs with a learning rate of 0.003 → 0.00001, three warmup epochs, and SWA over epochs 45–60.

Use **`runs/mild60/swa.pt`** for the reported submission. This route uses 460 training epochs in total.

The architecture has three stages of four pre-activation blocks, widths 56/112/224, 3 × 3 main convolutions, and 1 × 1 projection shortcuts. Training combines SGD with Nesterov momentum 0.9, weight decay 0.0005, RandAugment N = 2 / M = 7, Mixup alpha 0.2, CutMix alpha 1, total mixing probability 0.8, label smoothing 0.05, and EMA 0.999. The continuation adds color jitter 0.15, hue jitter 0.0375, grayscale probability 0.05, and blur probability 0.03 with a 3 × 3 kernel and sigma 0.1–1.0.

### B. Robust WRN with SWA

```powershell
python -m dlend.train --config configs/robust300.json --output runs/robust300
python -m dlend.train --config configs/robust60.json --output runs/robust60 --init-checkpoint runs/robust300/best.pt
```

This model uses widths 58/116/232 and SE blocks with ratio 0.25, totaling 4,948,626 parameters. The initial run trains for 300 epochs and selected EMA epoch 285 in my experiment. A 60-epoch continuation produces **`runs/robust60/swa.pt`**, the model associated with the final private score of 86.30%.

The robust recipe uses RandAugment N = 2 / M = 5, color jitter 0.2, grayscale probability 0.1, blur probability 0.05, Mixup alpha 0.4, CutMix alpha 1, mixing probability 0.6, and label smoothing 0. The route uses 360 training epochs in total.

### C. Refine the robust model for 50 epochs

Continue from route B's SWA checkpoint:

```powershell
python scripts/train_from_swa.py --config configs/robust_refine50.json --output runs/robust_refine50 --init-checkpoint runs/robust60/swa.pt --check-only
python scripts/train_from_swa.py --config configs/robust_refine50.json --output runs/robust_refine50 --init-checkpoint runs/robust60/swa.pt
```

This phase uses a learning rate of 0.0005 → 0.00001, zero warmup, and SWA start 38. My submission used **`runs/robust_refine50/best.pt`**, which selected EMA weights at epoch 29 and scored 87.30% on the private split. The full route uses 410 training epochs.

The SWA wrapper checks the source checkpoint, matching architecture, data split, and training records before starting the next phase. `--check-only` runs those checks; the second command starts training. The wrapper preserves the training loop and provides a portable replacement for the original machine-specific checkpoint registration.

## 4. Evaluate and generate a submission

The report's final comparisons use CPU FP32, two threads, batch size 64, and horizontal-flip averaging.

```powershell
python scripts/inference_cpu.py evaluate --threads 2 --checkpoint runs/mild60/swa.pt --data-root data/competition --tta flip --batch-size 64 --output runs/mild60/validation_flip.json
python scripts/inference_cpu.py predict --threads 2 --checkpoint runs/mild60/swa.pt --test-csv data/competition/test.csv --image-dir data/competition/test_images --path-column filename --sample-submission data/competition/sample_submission.csv --output submissions/mild60.csv --tta flip --batch-size 64
```

For routes B and C, substitute the checkpoint path listed above. Keep the default `--weights model`: that field already contains the selected raw, EMA, or SWA weights. Flip inference averages a checkpoint's predictions for the original image and its horizontal flip.

The prediction script writes `id,label` rows in the sample submission's order and checks the label range. Upload the resulting CSV through Kaggle. Kaggle's public and private scores are recorded in [the results summary](../results/official_results_public_summary.json); the evaluation command computes local validation accuracy.

The repository supplies the training code and logs. Model weights are produced by running the chosen route.

## 5. Resume an interrupted phase

Use the same configuration and output directory:

```powershell
python -m dlend.train --config configs/mild60.json --output runs/mild60 --resume runs/mild60/last.pt
```

A phase that started from SWA also resumes through `dlend.train --resume`. Use `--init-checkpoint` when starting a new continuation directory.

## 6. Run the results notebook

```powershell
python scripts/execute_notebook.py
```

You can also open [`notebooks/reproduction.ipynb`](../notebooks/reproduction.ipynb) in Jupyter and run all cells. It plots all 870 epochs from the five main phases, displays the saved scores, and runs CPU parameter-count and shape checks.

Training curves show the logged BF16 single-view raw and EMA results. The FP32/flip and SWA scores appear as separate evaluated endpoints. With Mixup and CutMix, `train_acc` measures agreement with the mixed targets; validation accuracy uses the original class labels.

## Other experiments

The [archive](../archive/README.md) maps all 79 experiments to their configurations, logs, initialization dependencies, and source variants. The five main configurations change the original data path to `data/competition`; their training values are preserved. See [source_mapping.json](../provenance/source_mapping.json) for the file mappings and hashes.
