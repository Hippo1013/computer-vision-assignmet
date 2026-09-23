# 机器视觉实验报告

采用“青蓝实验报告”XeLaTeX 模板，包含“实验设置”“结果分析”“总结”三部分：2.1 比较输入尺寸，2.2 比较两种模型的精度并讨论尺度信息的作用，第三部分总结学习方法的效果、补充深度信息的收获及主动多视角观测的后续方向。

- `main.tex`：标题、副标题、成员信息和章节入口。
- `experiment-settings.tex`：实验设置正文。
- `results-relative.tex`：2.1 正文，含深度图、误差表、耗时表和点云示意。
- `results-metric.tex`：2.2 正文，包含公平比较设计、指标分析与尺度信息的作用。
- `conclusion.tex`：第三部分总结。
- `prepare_results.py`：从实验输出生成 `figures/` 图片、两个 `relative-*-rows.tex` 表格文件、`model-comparison-table.tex` 及 `alignment-gap-table.tex`。
- `qinglan-report.sty`：青蓝样式。
- `report.pdf`：当前预览。

编译：

```bash
cd report
bash build.sh main.tex report.pdf
```

需要 XeLaTeX 和 latexmk。编译辅助文件自动放入系统临时目录并清理。
图表依据 `Depth-Anything-V2/outputs/` 的已有输出生成，不重新运行模型。更新实验结果后，先在 `mono_depth` 环境中运行 `python prepare_results.py`，再编译。图表生成需要 NumPy、Matplotlib、Pillow 和 Open3D。

深度图统一显示对齐后的米制深度，色阶为 0.1–10 m；点云选用 0001 的参考点云与 518 预测点云，两幅使用同一视角和比例。误差表与耗时表分别读取组内 `aligned/metrics.csv` 和 `prediction/inference_timing.json`，均值按 8 张图等权计算。正文中的数值分析在实验数据更新后也应核对。

2.2 的表格包含两个输入尺寸下的三组结果：Depth Anything V2 对齐后、Depth Anything V2 (Metric Depth) 对齐后及原始米制。生成时从六组 `metrics.csv` 计算均值并与 `outputs/summary/means.csv` 核对；原始米制结果与对齐结果分别解释，不作混合排名。

指标表的粗体由脚本自动生成：2.1 逐图及均值行比较同一指标在 518、280 下的最小值，耗时表比较两组耗时；2.2 在同一输入尺寸下比较两种模型对齐后的指标，原始米制行不参与该排名。依据未舍入数值判定，并列最优同时加粗；均值行保留底色，不再整行加粗数字。

对齐差距表以同尺寸 Depth Anything V2 对齐后的指标均值为基准，计算 `(米制深度指标均值 / 基准均值 - 1) × 100%`，展示原始米制到对齐后的变化，并加粗较小差距。这是指标均值的比值，不是逐图百分比的平均值；对齐前的跨处理条件差距不用于模型排名。
