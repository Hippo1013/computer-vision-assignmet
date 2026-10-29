# CIFAR-100 优化器与学习率实验

本阶段只执行题目第1–3项。ResNet18/50、知识蒸馏、交通标志训练等待阶段结果讨论后决定。

## 实验方案

`stage1.json` 按作业顺序定义五组 ResNet10：

| 作业步骤 | 实验内容 | 结果复用 |
|---|---|---|
| 第一步：基线复现 | SGD，固定学习率0.001 | 作为第二、三步的基线 |
| 第二步：优化器比较 | SGD与Adam，均固定学习率0.001 | SGD复用第一步，只新增Adam一组 |
| 第三步：SGD学习率分析 | 固定0.001/0.01/0.1，以及初始0.1、完成30/40轮后分别乘0.1 | 固定0.001复用第一步，新增另外三组 |

Adam仅运行0.001，不进行Adam学习率搜索。每组50轮。SGD动量0.9；两种优化器权重衰减0.0001；Adam betas=(0.9,0.999)、eps=1e-8。

从官方训练集每类500张固定抽50张作为验证集，合计45000训练、5000验证，种子20261029。五组使用相同初始权重、每轮数据顺序与增强随机种子。保留课程代码的小卷积入口、三次下采样和预处理。FP32，关闭TF32，cuDNN确定性模式；单种子不代表统计显著差异。

原始课程入口仍在 `cv_homework2/resnet-cifar100.py`。新增入口直接复用其ResNet10定义，增加独立验证集、指标、优化器配置和恢复功能，不采用原脚本的每轮测试集评估。

## Windows 运行

工作目录为 `E:\机器视觉\homework`，解释器优先为 `D:\Anaconda\envs\letter_recognition_train\python.exe`。依赖沿用该环境的torch/torchvision，并补充课程要求的pytorch-ignite、configargparse。数据通过SSH复制至`exp4/cv_homework2/data/cifar-100-python/`，不通过Git。

在仓库根目录执行：

```powershell
& 'D:\Anaconda\envs\letter_recognition_train\python.exe' -u exp4/train_stage1.py --preflight-only
& 'D:\Anaconda\envs\letter_recognition_train\python.exe' -u exp4/train_stage1.py
```

默认输出`exp4/outputs/stage1_20261029_five_runs/`。短试跑自动检查Adam三批训练、模型参数更新、检查点加载及学习率边界，并在256/128/64中选择可用批大小。正式五组统一采用该值。执行第二条命令会依次完成五组、冻结验证选优、每组最佳验证检查点的测试评估，以及选中配置的50000张全训练集50轮重训和末轮测试。不得依据测试结果重新选参。

同一命令再次运行从每组`last.pt`恢复；配置或训练源码变化时拒绝复用原输出目录。每轮完成才提交检查点，未完成的一轮重新运行；每轮独立设置数据随机种子。最佳验证准确率并列时选择较早轮次，组间并列遵循配置顺序。非有限损失写入失败记录并跳过该组。

## 结果与交付

每组保留`config.json`、`history.json`、`last.pt`、`best.pt`（验证组）及`test.json`，根目录包含数据哈希、环境与代码版本、划分索引、选优冻结结果及状态。测试预测也保留在本地JSON中。所有输出与权重由根目录忽略。

结果复制回本地后生成阶段表格和曲线：

```bash
python exp4/summarize_stage1.py exp4/outputs/stage1_20261029_five_runs
```

训练时间按GPU同步后的每轮训练时间累计，另记录含验证时间；检查点写盘不计入。训练损失按样本数加权，验证损失和准确率使用无增强验证集。最终重训与45000张对照组分开呈现。

## 阶段执行记录

2026-10-29：本地/GitHub/Windows 训练源码已同步；Windows直接拉取GitHub曾超时，后通过SSH传输Git bundle并快进到相同提交。exp1–3既有成果和中文命名最终PDF已同步，没有重跑旧实验；源码通过语法检查，三份PDF均为5页。

复用 `letter_recognition_train` 的 Python 3.10.20、torch 2.5.1+cu121、torchvision 0.20.1+cu121，补充pytorch-ignite 0.5.5与configargparse 1.9.0。三个CIFAR-100文件SHA-256在两端一致。分层划分、交集、重复生成及学习率边界检查通过。

Windows预检：batch=256通过，峰值张量分配1080732160字节；参数发生更新，检查点加载前后预测一致，恢复优化器后下一步更新与连续训练一致。原PowerShell日志重定向因普通弃用提示提前终止，改由Python启动器管理子进程日志；未改变训练参数。原七组方案曾启动后台任务 `MachineVision-Exp4-Stage1-20261029`，后已在SGD基线第7轮完成检查点后暂停。前两轮含验证耗时约51–52秒，发现加载开销需进一步优化；仅为运行检查，不作为最终结论。旧记录保留于`outputs/stage1_20261029/`。2026-10-29按用户修订改为当前五组方案，新输出使用独立目录，当前正式训练保持暂停。

后台任务运行 `run_stage1.ps1` / `run_stage1.py`，结束后自动生成阶段表格和三组曲线。可在Windows任务计划程序查看同名任务；手动恢复时重新运行启动器即可。不要并行启动第二个训练实例，入口具有输出目录锁。

本地通过 `python exp4/fetch_stage1.py` 拉回状态、记录与图表，通过 `python exp4/fetch_stage1.py --weights` 额外拉回权重，全程使用SSH。结果跟进已暂停，避免按旧七组配置自动恢复；后续恢复时采用五组方案，正常推进时保持安静，完成或失败时继续处理并汇报。

本阶段不撰写整份最终报告；五组对照、最终重训与阶段分析仍待完成。最终报告须按根目录规范自检且不超过5页。
