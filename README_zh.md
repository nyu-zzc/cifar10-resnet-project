# CIFAR-10 residual networks under 5M parameters

项目公开仓库：[nyu-zzc/cifar10-resnet-project](https://github.com/nyu-zzc/cifar10-resnet-project)。

可公开复现的课程项目代码与真实实验记录。核心模型从随机初始化训练，只使用比赛提供的数据；后续阶段从本项目自己的权重继续训练。比赛入口：[CIFAR-10 ResNet Architecture Challenge](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl)。

本仓库包括原始训练/预测源码、5 份便携配置、实际逐轮日志、已执行 notebook、79 项实验清单及结果摘要。**不包含比赛图像、逐样本标签、提交预测、账号信息或模型权重。** 数据须通过比赛的正式渠道自行取得。训练命令会进行真实长时间训练；查看 notebook 不会启动训练。

## 结果与模型身份

| 角色 | 模型 / 选中权重 | 参数 | 本地验证 FP32 + flip | Public | Private |
|---|---|---:|---:|---:|---:|
| 公榜最高，截图中已勾选 | WRN-28-3.5，mild60 SWA | 4,480,474 | 97.36% | 89.4% | 86.1% |
| 截图所示最终选择结果 | WRN-28-3.625 + SE，robust60 SWA | 4,948,626 | 97.40% | 88.6% | **86.3%** |
| 截图中观察到的私榜最高，未勾选 | 同一 SE 架构，robust_refine50 EMA29 | 4,948,626 | 97.48% | 88.6% | 87.3% |

Public/Private 来自 2026-11-09 用户提供的提交页截图，与本地同名 CSV 及权重记录核对；未独立下载核对已上传文件的字节。截图的五个唯一提交中，87.3% 属于未选中的提交，不能写作最终成绩。原截图包含账号界面，因此不公开附带；整理后的数值和身份边界见 [官方结果摘要](results/official_results_public_summary.json)。

本地验证为固定 5,000 张有标签图像；训练为 45,000 张，划分 seed=42；比赛隐藏测试为 2,000 张。验证成绩与两个榜单属于不同评估范围，不能直接互换。79 项实验共运行 7,810 个阶段 epoch（39 个筛选、18 个从零完整训练、22 个续训），见 [完整实验清单](results/experiment_ledger.json)。这不表示所有组合均已穷举或已证明全局收敛，也不代表每个方法均有效。

## 环境与安装

历史训练环境：Python 3.12.10、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、NumPy 2.5.4、CUDA 12.8、Windows 11、NVIDIA GeForce RTX 5070 Ti。版本属于已运行配置，不声称是当前最新版。建议新建 Python 3.12 环境，在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/verify_package.py
```

Linux/macOS 激活方式为 `source .venv/bin/activate`；仅查看 notebook 或 CPU 检查时，可从 PyTorch 官方源安装相同版本的 CPU wheel。真实训练配置保留原 `device=cuda`、BF16、batch128、workers2；调整为 CPU、其他精度或 worker 数均应明确记录为环境/复现变化。训练设置 `deterministic=false`，重跑不承诺逐位相同、相同最佳 epoch 或相同榜单成绩。

## 数据准备

```powershell
python scripts/prepare_competition.py --archive data/competition.zip --output data/competition
```

比赛 ZIP 需要包含原 `train_labels.csv`（id,filename,label）、`test.csv`（id,filename）、`sample_submission.csv`（id,label）和对应 32×32 RGB 图像。脚本生成 `train.npz`、`test_images/`、三个 CSV 和审计摘要；训练图像直接从 ZIP 解码，不必解压为 50,000 个文件。代码不下载其他数据。提交模板中的占位标签不用于训练或评估。

类别索引 0–9 依次为 airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck；模型输出编码保持字符串 "0"–"9"。训练/验证内容指纹及聚合计数见 [数据摘要](results/data_summary.json)。相同内容的训练样本划在同一侧；原数据审计没有发现训练/测试完全相同像素。

## 复现路线 A：公榜最佳 mild appearance

```powershell
python -m dlend.train --config configs/base400.json --output runs/base400
python -m dlend.train --config configs/mild60.json --output runs/mild60 --init-checkpoint runs/base400/best.pt
```

首阶段从零训练 400 epochs，原实验的 `best.pt` 是第 396 轮原始权重；续训重启优化器、EMA、随机状态和学习率日程。第二阶段 60 epochs，lr=0.003→0.00001，warmup3，SWA 对第 45–60 轮原始权重平均后用训练数据重估 BN。最终用于此结果的是 **runs/mild60/swa.pt**，不是其 `best.pt`。实际总计算为 400+60=460 epochs；选中权重的来源是396+60。

WRN 使用 3×3 stem（3→16），三个残差阶段每阶段 4 个预激活 basic block，每块两个3×3卷积，通道56/112/224，第二/三阶段首块stride2，1×1投影 shortcut，最终BN-ReLU、global average pooling和10类线性层。没有SE、DropPath或dropout。28是WRN的深度命名惯例，不应误写为28个basic blocks。

基础增强为反射填充4的随机裁剪、水平翻转、RandAugment N=2/M=7；mild60 额外设置 ColorJitter=0.15、随机灰度p=0.05、高斯模糊p=0.03（kernel3，sigma0.1–1.0）。Mixup alpha0.2、CutMix alpha1.0、混合总概率0.8，每次选择一种，label smoothing0.05。两阶段都使用SGD momentum0.9/Nesterov、weight decay0.0005、EMA0.999。

## 复现路线 B：最终选择 robust SWA

```powershell
python -m dlend.train --config configs/robust300.json --output runs/robust300
python -m dlend.train --config configs/robust60.json --output runs/robust60 --init-checkpoint runs/robust300/best.pt
```

该分支宽度3.625，通道58/116/232，SE比例0.25，总参数4,948,626。原从零300轮的最佳为EMA285，再续训60轮并选择 **runs/robust60/swa.pt**。总计算360轮，选中权重谱系285+60=345轮。

与路线A的关键区别：RandAugment M=5，ColorJitter0.2、灰度p0.1、模糊p0.05；Mixup alpha0.4、CutMix alpha1、混合概率0.6、label smoothing0。其余精确数值在配置中。不同架构和增强同时改变，不能将跨路线差异归因于某一单独组件。

## 复现路线 C：观察到的私榜最高（未选中）

完成路线B后，从其原始 SWA 继续训练：

```powershell
python scripts/train_from_swa.py --config configs/robust_refine50.json --output runs/robust_refine50 --init-checkpoint runs/robust60/swa.pt --check-only
python scripts/train_from_swa.py --config configs/robust_refine50.json --output runs/robust_refine50 --init-checkpoint runs/robust60/swa.pt
```

该阶段50轮，lr0.0005→0.00001、warmup0、SWA起点38；实际提交使用 **runs/robust_refine50/best.pt** 中的EMA29，而不是SWA。总计算300+60+50=410轮；选中谱系285+60+29=374轮。

`train_from_swa.py` 是公开包新增的便携准入包装。它核对原始 `swa.pt`、SWA自身指标、连续完整日志、数据/划分指纹、架构和项目随机起源声明；加载同一份模型与BN权重，随后调用逐字复制的原训练循环。可用 `--expected-source-sha256 HEX` 进一步固定来源。历史运行的准入还绑定私有机器上的计划与归档哈希；这些私人路径未公开复制，准入方式的差别不改变训练数学。此包装不能证明任意第三方伪造元数据的权重具有可信历史，请只加载自己训练或可信来源的项目权重。

## 评估和生成提交

下列便携包装明确使用CPU FP32与2个线程；原evaluate会按checkpoint将线程改回4，仅环境变量不能保证2线程。包装只改运行时线程设置，不改权重与推理数学；训练中的BF16单视图日志不可直接当作FP32+TTA结果。原型重训没有附带权重，因此必须先完成对应路线或自行取得可信的原始权重。

```powershell
python scripts/inference_cpu.py evaluate --threads 2 --checkpoint runs/mild60/swa.pt --data-root data/competition --tta flip --device cpu --batch-size 64 --output runs/mild60/validation_flip.json
python scripts/inference_cpu.py predict --threads 2 --checkpoint runs/mild60/swa.pt --test-csv data/competition/test.csv --image-dir data/competition/test_images --path-column filename --sample-submission data/competition/sample_submission.csv --output submissions/mild60.csv --tta flip --device cpu --batch-size 64
```

路线B/C将checkpoint替换为上文对应路径。`best.pt` 内的 `model` 已是选中的EMA或原始权重，`swa.pt` 内的 `model` 是SWA权重；都使用默认 `--weights model`。flip为原图与水平翻转预测概率的平均，不是两个不同模型的集成；参数量不增加。预测输出严格遵循样例提交ID顺序，不会自动上传。

训练中断后，在相同配置、相同目录运行 `python -m dlend.train --config configs/mild60.json --output runs/mild60 --resume runs/mild60/last.pt`。SWA包装启动的阶段也用同样的 `dlend.train --resume` 机制恢复，不能再次以 `--init-checkpoint` 覆盖已有阶段。

## Notebook、验证与复现边界

打开 [notebooks/reproduction.ipynb](notebooks/reproduction.ipynb)。它读取本仓库真实日志，展示全部相关轮次曲线、三种结果身份和CPU参数审计；不训练、不推理比赛样本。已保存输出通过 `scripts/execute_notebook.py` 按单元顺序执行，执行方式会写入notebook元数据。也可以用Jupyter的Run All重新执行。

```powershell
python scripts/execute_notebook.py
python scripts/verify_package.py
python -m dlend.train --help
python -m dlend.predict --help
```

`train_acc` 在Mixup/CutMix下是soft-target agreement，不是普通干净训练集准确率；曲线不把它与验证准确率直接比较。SWA保留cosine末段日程，属于SWA启发的实现，不是论文constant/cyclic LR配方严格复现。单次验证上的多轮选模和多个方法比较有选择偏差，不能据此证明私榜必然提高。

`provenance/source_mapping.json` 给出主要源码/聚合日志的原始与公开SHA，以及配置路径映射。5份主要配置只把 `dataset.root` 从原工作区相对位置改为 `data/competition`，其他数值不变。[完整归档](archive/README.md) 进一步包含79项实验的完整便携配置、7,810轮聚合日志、初始化依赖，以及对应的五套隔离源码快照；逐项映射见 `archive/inventory.json`。Efficient ResNets第三方源码的原MIT notice完整保留。原机器私有的队列授权/准入记录没有公开，不能声称全部归档命令都已在新包中重新运行。

大权重暂不发布；因此本包提供可运行的主要从零复现流程和已运行证据，不伪称无需重训即可重现原始预测字节。公开包重点验证上述三个提交身份的5份阶段配置；其他源码与配置用于完整追溯及进一步复现，不会自动启动所有历史队列。

代码许可证见 [LICENSE](LICENSE)，方法与依赖出处见 [REFERENCES.md](REFERENCES.md) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。MIT适用于本项目原创代码，不替代数据或第三方软件的许可证。
