# 方法出处与实现边界

下列条目为已核对的一手论文或官方文档。它们说明方法来源，不代表这些方法在本次比赛中已经有效。报告只保留实际使用的方法，并将实验结果与文献结果分开。

| 实现/候选方法 | 一手出处 | 本项目使用方式 |
|---|---|---|
| 残差连接 | He, K., Zhang, X., Ren, S. & Sun, J. *Deep Residual Learning for Image Recognition*. CVPR (2016). [论文](https://arxiv.org/abs/1512.03385) | 满足残差网络设计要求 |
| 预激活残差块 | He, K., Zhang, X., Ren, S. & Sun, J. *Identity Mappings in Deep Residual Networks*. ECCV (2016). [论文](https://arxiv.org/abs/1603.05027) | BN/ReLU 置于卷积前的残差块 |
| Wide ResNet | Zagoruyko, S. & Komodakis, N. *Wide Residual Networks*. BMVC (2016). [论文](https://arxiv.org/abs/1605.07146) | 在参数预算内调节深度和宽度；不是下载预训练权重 |
| SE 通道注意力 | Hu, J., Shen, L. & Sun, G. *Squeeze-and-Excitation Networks*. CVPR (2018). [论文及修订记录](https://arxiv.org/abs/1709.01507) | 可选模块；加入后重新计算参数量；arXiv 后续期刊版增加作者 |
| Stochastic depth | Huang, G., Sun, Y., Liu, Z., Sedra, D. & Weinberger, K. *Deep Networks with Stochastic Depth*. ECCV (2016). [论文](https://arxiv.org/abs/1603.09382) | 可选按样本 DropPath；属于实现变体，不能把原文节省计算的结论直接套用 |
| RandAugment | Cubuk, E. D., Zoph, B., Shlens, J. & Le, Q. V. *RandAugment: Practical automated data augmentation with a reduced search space*. CVPR Workshops (2020). [论文](https://arxiv.org/abs/1909.13719) | 验证集选择增强强度；不额外搜索外部数据 |
| AutoAugment | Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V. & Le, Q. V. *AutoAugment: Learning Augmentation Policies from Data*. CVPR (2019). [论文](https://arxiv.org/abs/1805.09501) | 可选 CIFAR-10 policy；与 RandAugment 比较而非同时启用 |
| Mixup | Zhang, H., Cisse, M., Dauphin, Y. N. & Lopez-Paz, D. *mixup: Beyond Empirical Risk Minimization*. ICLR (2018). [论文](https://arxiv.org/abs/1710.09412) | 图像和软标签线性混合 |
| CutMix | Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J. & Yoo, Y. *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*. ICCV (2019). [论文](https://arxiv.org/abs/1905.04899) | 标签权重按实际裁剪区域修正 |
| Cutout | DeVries, T. & Taylor, G. W. *Improved Regularization of Convolutional Neural Networks with Cutout*. arXiv (2017). [论文](https://arxiv.org/abs/1708.04552) | 可选遮挡；与强 CutMix/Random Erasing 组合前做消融 |
| Random Erasing | Zhong, Z., Zheng, L., Kang, G., Li, S. & Yang, Y. *Random Erasing Data Augmentation*. arXiv (2017). [论文](https://arxiv.org/abs/1708.04896) | 可选随机擦除 |
| Label smoothing | Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. & Wojna, Z. *Rethinking the Inception Architecture for Computer Vision*. CVPR (2016). [论文](https://arxiv.org/abs/1512.00567) | 轻度平滑；Mixup/CutMix 已软化标签，需验证是否过度正则化 |
| Cosine 学习率 | Loshchilov, I. & Hutter, F. *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR (2017). [论文](https://arxiv.org/abs/1608.03983) | 使用 warmup + 无重启 cosine；不声称完整复现 SGDR |
| AdamW | Loshchilov, I. & Hutter, F. *Decoupled Weight Decay Regularization*. ICLR (2019). [论文](https://arxiv.org/abs/1711.05101) | SGD 的可选对照，学习率应独立调整 |
| SAM | Foret, P., Kleiner, A., Mobahi, H. & Neyshabur, B. *Sharpness-Aware Minimization for Efficiently Improving Generalization*. ICLR (2021). [论文](https://arxiv.org/abs/2010.01412) | 两次前后向增加训练成本；独立对照 |
| 知识蒸馏 | Hinton, G., Vinyals, O. & Dean, J. *Distilling the Knowledge in a Neural Network*. arXiv (2015). [论文](https://arxiv.org/abs/1503.02531) | 只能使用获准数据从零训练的本项目教师；保持训练/验证划分一致 |
| EMA | [PyTorch 官方权重平均文档](https://docs.pytorch.org/docs/stable/optim.html#weight-averaging-swa-and-ema) | 比较原始权重与 EMA；推理只加载选中的一份模型 |
| SWA | Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D. & Wilson, A. G. *Averaging Weights Leads to Wider Optima and Better Generalization*. UAI (2018). [论文](https://arxiv.org/abs/1803.05407) | 训练后期权重平均并重新估计 BN；本实现保留 cosine 日程，属于 SWA 启发的变体，并非原文 constant/cyclic LR 配方复现 |
| AMP | [PyTorch 官方 AMP 文档](https://docs.pytorch.org/docs/stable/amp.html) | 加速选项；不视为准确率提升证据 |


Environment reference: [PyTorch historical installation matrix](https://pytorch.org/get-started/previous-versions/). The public package pins the versions actually used; it does not claim they are current.
