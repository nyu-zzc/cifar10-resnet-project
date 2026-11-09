# Full experimental configuration and source archive

`inventory.json` maps all **79 completed tasks / 7,810 epochs** to the complete runtime configuration, original aggregate per-epoch logs, model parameter count, initialization dependency and exact source variant. Only dataset/checkpoint path strings are relocated; numerical configuration values remain unchanged. Original and published hashes are listed separately when paths changed. No sample image, label array, prediction row or weight file is distributed.

Each entry identifies its `pythonpath`. The main six queues use the base `src`; later research branches used isolated package copies. Five source snapshots are bundled under `source_variants/`: architecture completion, single-view AugMix, champion continuation, Efficient ResNets, and four-stage WRN. Source files are byte-identical copies with hashes. Configuration values alone do not identify the implementation; always use the entry's module variant.

Example for the from-scratch K3 shortcut experiment (PowerShell, run from repository root):

```powershell
$env:PYTHONPATH = (Resolve-Path archive/source_variants/architecture_completion_v1/src).Path
python -m dlend.train --config archive/configs/architecture_completion_v1/shortcut_kernel3_200.json --output runs/architecture_completion_v1/runs/shortcut_kernel3_200
Remove-Item Env:PYTHONPATH
```

For a warm-start entry, first reproduce its `initialization.checkpoint` dependency, then add `--init-checkpoint` with that path; keep the source `best.json`. For knowledge distillation, reproduce the teacher path recorded in the configuration. Using the entry's `original_project_run_relative` as the output path preserves these relative dependencies. Screens are intentionally shorter than final runs and cannot be reported as equal-budget comparisons to longer runs.

The original SWA-source admission in the champion snapshot requires a machine-local signed registration that is not publicly bundled. The root README supplies a portable checked-admission route for the report's robust refinement model. Other archived configurations are provided with exact values and source mappings, but the public package has not re-executed every historical training command. No automated queue is launched by opening the archive or notebook.

`efficient_resnets_v1/src/dlend/vendor_project1_model.py` is the supplied author's architecture. Its original MIT notice is preserved in `source_variants/efficient_resnets_v1/LICENSE.vendor`: Copyright (c) 2017 liukuang. The corresponding paper is Aditya Thakur, Harish Chauhan and Nikunj Gupta, *Efficient ResNets: Residual Network Design*, arXiv:2306.12100. [Paper](https://arxiv.org/abs/2306.12100), [author repository](https://github.com/Nikunj-Gupta/Efficient_ResNets). No supplied pretrained checkpoint was loaded for these project experiments, and no author checkpoint is distributed here.

SWA/best/final evaluations, TTA, post-training quantization and cached audits are artifacts or evaluations, not additional training tasks. Model selection on the same validation data and hardware/precision differences limit claims of improvement; the archive records completion, not proof of universal effectiveness or convergence.
