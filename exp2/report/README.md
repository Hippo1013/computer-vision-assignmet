# 机器视觉实验报告

Git 保留手写 LaTeX 源码、样式、生成脚本和最终 `report.pdf`。`figures/` 和自动生成的表格只保留在本地。全新克隆后可直接查看现成 PDF；如需重新编译，应先按实验指令准备数据和权重、运行实验，再执行 `prepare_results.py` 生成图表。

使用青蓝实验报告 XeLaTeX 模板，标题结构与实验1一致，正文为“实验设置”“结果分析”“总结”。报告限五页，包含输入视图数比较、点云阈值比较、反投影公式推导和后处理前后展示。

- `main.tex`：标题、作者、章节入口。
- `experiment-settings.tex`：数据、模型、实验分组与统计约定。
- `results-views.tex`、`view-analysis.tex`：视图数比较及分析。
- `results-fusion.tex`、`fusion-analysis.tex`：点云阈值比较及分析。
- `geometry-analysis.tex`：相机与世界坐标变换、单视角点云对比。
- `conclusion.tex`：多视角点云对比、总结与后续思考。
- `prepare_results.py`：从正式实验输出生成图表。
- `qinglan-report.sty`、`build.sh`：青蓝样式和临时目录编译脚本。
- `report.pdf`：最终五页报告。

在 `exp2` 目录中执行：

```bash
conda activate mvsnet
python report/prepare_results.py
bash report/build.sh main.tex report.pdf
```

图表只读取 `outputs/views3/`、`outputs/views5/` 的正式结果。深度图统一使用 425–935 mm 色阶；置信度图统一使用 0–1，超过 1 的上游置信度值仅在显示时饱和为白色，原始统计不裁剪。点云采用同一正交投影方向，阈值对比使用同一包围范围，直接/后处理比较则共享原始点云的完整包围范围。渲染使用最近点遮挡判断，不删掉不美观的点来修饰结果。

表格由数组及 CSV/JSON 汇总生成；正文的分析数值需要在实验更新后同步核对。置信度和点云保留比例没有真实精度含义。体素合并后的点数与过滤前后观测点数分开记录，避免混淆。

此报告包含样式、章节和图片等多个本地文件。最终 PDF 使用本机 XeLaTeX/latexmk 编译；内置编辑器可用于编辑入口文件，但当前内置编译器不加载其他项目文件，会提示找不到 `qinglan-report.sty`；请以 `build.sh` 生成的 PDF 为准。编译辅助文件只在唯一系统临时目录产生，结束时自动删除。
