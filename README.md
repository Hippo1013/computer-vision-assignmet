# computer-vision-assignmet

机器视觉课程小组实验作业仓库，按每周实验目录共同管理代码、实验资料、数据和结果。

## 当前内容

- `exp1/实验课1-单目深度与点云重建.pdf`：实验要求。
- `exp1/Depth-Anything-V2/`：Depth Anything V2 源码、本地实验脚本、模型权重、输入数据和实验输出。
- 后续实验可按 `exp2/`、`exp3/` 等目录继续添加。

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

实验资料、模型权重、数据与已有输出均保留；忽略 macOS 元数据和 Python 字节码缓存。仓库包含大体积模型及结果，首次克隆可能需要一些时间。

`exp1/Depth-Anything-V2/` 以实际文件纳入本仓库，无需初始化子模块。上游项目为 [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)，其说明和许可证保留在原目录中；模型的使用条件请同时参阅对应上游说明。

若使用创建本仓库的原始本地工作区，请在本仓库根目录执行小组协作的 Git 命令；其 `Depth-Anything-V2` 子目录仍保有原先的独立 Git 记录。
