# computer-vision-assignmet

机器视觉课程小组实验作业仓库，按每周实验目录管理程序源码、配置、运行说明、可编辑的 LaTeX 报告源码和最终报告 PDF。数据、权重和实验输出只保留在本地。

## 当前内容

- `exp1/Depth-Anything-V2/`：Depth Anything V2 源码与本地实验脚本；入口见 `exp1/实验指令汇总.md`。
- `exp2/`：MVSNet 多视图深度、点云融合、参数对比脚本与青蓝报告源码；入口见 `exp2/实验指令汇总.md`。
- 后续实验可按 `exp3/` 等目录继续添加。

## 小组协作

```bash
git clone https://github.com/Hippo1013/computer-vision-assignmet.git
cd computer-vision-assignmet
git switch -c exp1/your-name
# 修改或添加本周实验文件后，在仓库根目录运行：
git add .
git commit -m "exp1: describe your changes"
git push -u origin exp1/your-name
```

在 GitHub 创建 Pull Request，由队友检查后合并到 `main`。开始新的实验分支前，先切回 `main` 并执行 `git pull --ff-only`。

私有仓库需要先在仓库的 **Settings → Collaborators** 中邀请队友，队友接受邀请后即可克隆和推送。

## 文件与来源

远端只维护源码、必要配置、许可证和说明，包括手写的 `.tex`、`.sty` 报告源码及两份最终 `report/report.pdf`。模型权重、输入数据、课程课件、预测结果、点云、日志、生成图表与表格均由 Git 忽略；最终报告 PDF 作为交付文件保留。已有本地文件仍在原路径，取消跟踪不会删除它们。

仓库历史也已清除图片、点云等本地实验产物，保留最终报告 PDF。历史重写前的旧克隆请重新克隆，避免将旧历史重新推回远端。

新克隆的仓库需要另行准备各实验的数据和权重；按对应的 `实验指令汇总.md` 运行后会生成结果。最终报告 PDF 可直接查看；若需重新编译报告，先运行实验和 `report/prepare_results.py`，再编译。

`exp1/Depth-Anything-V2/` 以实际文件纳入本仓库，无需初始化子模块。上游项目为 [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)，其说明和许可证保留在原目录中；模型的使用条件请同时参阅对应上游说明。

若使用创建本仓库的原始本地工作区，请在本仓库根目录执行小组协作的 Git 命令；其 `Depth-Anything-V2` 子目录仍保有原先的独立 Git 记录。
