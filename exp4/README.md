# CIFAR-100 优化器与学习率实验

本阶段只执行题目第1–3项。ResNet18/50、知识蒸馏、交通标志训练等待阶段结果讨论后决定。

## 实验方案

`stage1.json` 定义七组 ResNet10：SGD 固定学习率 0.001/0.01/0.1，Adam 固定学习率 0.0001/0.001/0.01，SGD 初始0.1并在完成第30、40轮后分别乘0.1。每组50轮。SGD动量0.9；两种优化器权重衰减0.0001；Adam betas=(0.9,0.999)、eps=1e-8。

从官方训练集每类500张固定抽50张作为验证集，合计45000训练、5000验证，种子20261029。七组使用相同初始权重、每轮数据顺序与增强随机种子。保留课程代码的小卷积入口、三次下采样和预处理。FP32，关闭TF32，cuDNN确定性模式；单种子不代表统计显著差异。

原始课程入口仍在 `cv_homework2/resnet-cifar100.py`。新增入口直接复用其ResNet10定义，增加独立验证集、指标、优化器配置和恢复功能，不采用原脚本的每轮测试集评估。

## Windows 运行

工作目录为 `E:\机器视觉\homework`，解释器优先为 `D:\Anaconda\envs\letter_recognition_train\python.exe`。依赖沿用该环境的torch/torchvision，并补充课程要求的pytorch-ignite、configargparse。数据通过SSH复制至`exp4/cv_homework2/data/cifar-100-python/`，不通过Git。

在仓库根目录执行：

```powershell
& 'D:\Anaconda\envs\letter_recognition_train\python.exe' -u exp4/train_stage1.py --preflight-only
& 'D:\Anaconda\envs\letter_recognition_train\python.exe' -u exp4/train_stage1.py
```

默认输出`exp4/outputs/stage1_20261029/`。短试跑自动检查Adam三批训练、模型参数更新、检查点加载及学习率边界，并在256/128/64中选择可用批大小。正式七组统一采用该值。执行第二条命令会依次完成七组、冻结验证选优、每组最佳验证检查点的测试评估，以及选中配置的50000张全训练集50轮重训和末轮测试。不得依据测试结果重新选参。

同一命令再次运行从每组`last.pt`恢复；配置或训练源码变化时拒绝复用原输出目录。每轮完成才提交检查点，未完成的一轮重新运行；每轮独立设置数据随机种子。最佳验证准确率并列时选择较早轮次，组间并列遵循配置顺序。非有限损失写入失败记录并跳过该组。

## 结果与交付

每组保留`config.json`、`history.json`、`last.pt`、`best.pt`（验证组）及`test.json`，根目录包含数据哈希、环境与代码版本、划分索引、选优冻结结果及状态。测试预测也保留在本地JSON中。所有输出与权重由根目录忽略。

结果复制回本地后生成阶段表格和曲线：

```bash
python exp4/summarize_stage1.py exp4/outputs/stage1_20261029
```

训练时间按GPU同步后的每轮训练时间累计，另记录含验证时间；检查点写盘不计入。训练损失按样本数加权，验证损失和准确率使用无增强验证集。最终重训与45000张对照组分开呈现。

当前状态：训练入口已实现，正式训练尚未开始。此阶段不撰写最终报告；最终报告仍须按根目录自检规范执行且不超过5页。
