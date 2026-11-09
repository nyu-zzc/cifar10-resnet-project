# Attribution and scope of license

The residual network, data adapter and training loop in `src/dlend/` are the project's original implementation of published methods. No pretrained weights or competition data are bundled.

The historical archive includes the supplied Efficient ResNets architecture in `archive/source_variants/efficient_resnets_v1/src/dlend/vendor_project1_model.py`. Its original MIT license is preserved, without replacing the copyright holder, at `archive/source_variants/efficient_resnets_v1/LICENSE.vendor` (Copyright (c) 2017 liukuang). Corresponding paper: Aditya Thakur, Harish Chauhan and Nikunj Gupta, *Efficient ResNets: Residual Network Design*, [arXiv:2306.12100](https://arxiv.org/abs/2306.12100); [author repository](https://github.com/Nikunj-Gupta/Efficient_ResNets). The external checkpoint supplied alongside that code was not loaded. Project adapters and the vendor architecture are distinguished in the archive.

PyTorch, torchvision, NumPy, Pillow, matplotlib, Jupyter and their dependencies retain their own licenses. They are installed as dependencies, not relicensed by this repository's MIT license. torchvision supplies the implementations of RandAugment, AutoAugment and image transforms; PyTorch supplies optimizers, AMP and SWA averaging/BN updates. See each installed distribution's license and the linked official projects.

- [PyTorch](https://github.com/pytorch/pytorch)
- [torchvision](https://github.com/pytorch/vision)
- [NumPy](https://github.com/numpy/numpy)
- [Pillow](https://github.com/python-pillow/Pillow)
- [matplotlib](https://github.com/matplotlib/matplotlib)
- [Jupyter](https://jupyter.org/)

Method citations are listed in `REFERENCES.md`; citation of a paper is not a claim to have strictly reproduced its published recipe or results. Model weights and competition data are not covered by a redistribution grant in this package.
