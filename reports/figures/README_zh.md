# 报告图表与源数据

这里保存了报告的四张图、五份聚合数据表、图注和绘图脚本。每张图都有三种格式：SVG 方便编辑文字和图形，PDF 适合排版，PNG 适合放进 Word 或直接预览。

| 报告顺序 | 内容 | 文件名 |
|---|---|---|
| Figure 1 | 训练曲线 | `fig1_training_curves` |
| Figure 2 | 架构比较 | `fig3_architecture_comparison` |
| Figure 3 | 分类错误与模糊实验 | `fig4_errors_and_robustness` |
| Figure 4 | 本地验证与 Kaggle 成绩 | `fig2_official_scores` |

文件名保留了最初绘图时的编号，图注与当前报告的顺序一致。

## 修改和重新绘图

在本目录打开终端，使用 Python 3.12 环境：

```text
python -m pip install -r requirements.txt
python plot.py
```

脚本读取目录中的 CSV 并重新生成图片。调整颜色、字号或版式时，编辑 `plot.py` 即可；运行前可以备份自己手工编辑过的图。`sources.json` 记录了数据表的行数和哈希，便于核对绘图输入。

训练曲线来自逐轮日志，保留了原始波动。架构表使用统一验证设置下的结果。成绩图把本地、公榜和私榜放在一起，便于查看模型排名的变化；最终选定提交的私榜成绩是 86.30%。
