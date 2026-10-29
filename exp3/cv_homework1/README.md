# HOG 特征与 SVM 图像分类实验

## 环境配置

本项目使用 Conda 环境 `svm_classifier`，Python 3.11。依赖包括 NumPy、SciPy、scikit-image、scikit-learn、Matplotlib、OpenCV 和 joblib。

当前设备已创建环境，进入代码目录后激活：

```bash
cd "/Users/hippo/Desktop/courses/机器视觉/homework/exp3/cv_homework1"
conda activate svm_classifier
```

其他设备可在本目录使用环境配置重新创建：

```bash
conda env create -f environment.yml
conda activate svm_classifier
```

IDE 的 Python 解释器选择 `/Users/hippo/miniconda3/envs/svm_classifier/bin/python`。

## 实验入口

请从本目录运行程序，相对路径以本目录为基准。

| 程序 | 用途 | 本地输入 |
| --- | --- | --- |
| `visualize_hog.py` | HOG 特征可视化 | `jinx.jpg` |
| `compare_hog_parameters.py` | 要求2的参数对比、完整特征与中间直方图保存、结果审阅 | `jinx.jpg` |
| `verify_hog_dimension.py` | 实验要求第三点的 HOG 维度验证 | `jinx.jpg` |
| `extractFeat.py` | CIFAR-10 特征提取 | `cifar-10-batches-py/` |
| `classifier.py` | CIFAR-10 线性 SVM 训练与测试 | `data/features/old/` 或 `data/features/new/` 下的 `train/`、`test/` |
| `extractFeat_TrafficSign.py` | 交通标志 HOG 特征提取 | `Traffic_sign/train_dataset/`、`Traffic_sign/test_dataset/` |
| `classifier_TrafficSign.py` | 交通标志线性 SVM 训练与测试 | `outputs/traffic_sign/old/` 或 `new/` |

HOG 提取代码已补全。`visualize_hog.py` 中第一点的灰度图和直方图保存代码当前被注释。完整分类流程可依次运行：

```bash
python visualize_hog.py
python extractFeat.py
python classifier.py
```

环境安装不包含完整数据集的特征提取与训练。数据、生成特征和图表仅保留在本地。

## CIFAR-10分类

要求2的独立参数对比见下方“HOG参数可视化与特征审阅”，无需重新运行分类实验。

`extractFeat.py` 和 `classifier.py` 均支持 `--params old/new`，默认使用 `old`。

| 参数组 | orientations | pixels_per_cell | cells_per_block | 特征维度 | SVM C |
| --- | --- | --- | --- | --- | --- |
| `old`：原始参数 | 9 | (8,8) | (2,2) | 324 | 1 |
| `new`：调优参数 | 9 | (4,4) | (4,4) | 3600 | 0.2 |

```bash
# 默认旧参数，也可显式增加 --params old
python extractFeat.py
python classifier.py

# 新参数
python extractFeat.py --params new
python classifier.py --params new
```

输入仍为32×32 CIFAR-10图像，灰度化、`/255.0` 与 HOG 归一化方式不变。两组特征分别保存在 `data/features/old/{train,test}/` 与 `data/features/new/{train,test}/`，保留原有逐图 `.feat` 格式；分类器检查参数清单、样本数量及特征维度，并在终端输出训练／测试准确率与收敛状态。两组SVM均用 `max_iter=1000`，仅C分别为1和0.2，固定随机种子20261022。

两脚本都可用 `--feature-root` 指定包含 `train/`、`test/` 的特征根目录；提取器另支持 `--data-dir`。此前直接位于 `data/features/train/`、`test/` 的特征保持原样。需要复用其中已有的3600维调优特征时，可执行 `python classifier.py --params new --feature-root data/features`，无需重新提取。导出的 `hog`、`rgb2gray`、`unpickle` 接口保持兼容；历史调优记录及其原有哈希校验不改写。

## 交通标志分类

第五点使用独立脚本，与 CIFAR-10 的分类入口及产物分开。交通标志训练集6024张、测试集324张，共10类；类别由目录名确定，并共用一份按名称排序的整数标签映射。以下 old/new 迁移比较沿用数据集现有训练／测试划分，不增加验证集或重新调参；后续固定 C=1 的独立 HOG 调优见下一节。

两组均直接使用224×224原尺寸 RGB 图像，不缩放；沿用原脚本灰度化系数 `0.2989/0.5870/0.1140` 与 `/255.0`，使用 `L2-Hys`、不加增强或标准化。

| 参数组 | orientations | pixels_per_cell | cells_per_block | 特征维度 | SVM C |
| --- | --- | --- | --- | --- | --- |
| `old`：原始参数 | 9 | (8,8) | (2,2) | 26244 | 1 |
| `new`：CIFAR-10调优参数 | 9 | (4,4) | (4,4) | 404496 | 0.2 |

SVM 均为 `LinearSVC(max_iter=1000, tol=1e-4, dual="auto", loss="squared_hinge", penalty="l2", random_state=20261022)`。随机种子用于复现，其他设置与原两脚本及此前调优配置对应；训练未收敛时程序报错，不将其记录为完成结果。新参数是此前 CIFAR-10 的调优结果，本次仅比较迁移效果，不称为交通标志最优参数。

```bash
python extractFeat_TrafficSign.py --params old
python classifier_TrafficSign.py --params old
python extractFeat_TrafficSign.py --params new
python classifier_TrafficSign.py --params new
```

省略 `--params` 时默认 `new`。输入与输出默认相对于脚本所在目录解析；可通过 `--dataset-root` 指定数据集位置，通过 `--output-root` 为两脚本指定同一个输出根目录。特征提取器不覆盖非空参数组目录，重复提取时应指定新的输出根目录，例如两脚本都增加 `--output-root outputs/traffic_sign_repeat`；仅重训或重放可直接运行分类器。

两组分别保存在 `outputs/traffic_sign/old/` 与 `outputs/traffic_sign/new/`。`train_features.npy`、`test_features.npy` 保存 float64 特征矩阵，逐张写入磁盘映射文件；对应 `train_labels.npz`、`test_labels.npz` 保存整数标签 `y` 和完整相对图像路径 `paths`，避免同名图片覆盖。分类器以磁盘映射方式读取特征，分批校验与预测，SVM仍用全部训练图像一次拟合。`manifest.json` 保存类别映射、样本路径与图像哈希、预处理、特征和标签哈希及依赖版本。分类器读取前校验特征与清单，输出 `model.joblib`、`results.json`、`predictions.npz`、`test_predictions.csv` 和 `summary.txt`；保存模型后重新加载核对全部训练／测试预测。终端自动输出训练与测试准确率。`results.json` 包含总体准确率、各类别指标、宏平均F1、混淆矩阵、训练耗时与收敛状态。这些产物仅保留在本地。

224×224原图的完整训练／测试结果：

| 参数组 | 训练准确率 | 测试准确率 | 测试正确数 | 测试宏平均F1 | SVM训练耗时 |
| --- | --- | --- | --- | --- | --- |
| `old` | 100.00% | 93.52% | 303/324 | 0.8930 | 19.91秒 |
| `new` | 100.00% | 95.06% | 308/324 | 0.9062 | 492.69秒 |

新参数测试准确率提高1.54个百分点，净多分对5张；两组均在1000次迭代上限内收敛，实际最大迭代次数分别为132和60。耗时为本机实际墙钟时间，受内存和换页影响。对比结果在 `outputs/traffic_sign/comparison.md`、`comparison.json`，核验记录在 `verification.json`，终端输出在 `old_run.log`、`new_run.log`。本次同时更换 HOG 和 C，结果反映两套参数组合的迁移效果。仅保留原尺寸实验产物。

## 交通标志 HOG 粗粒度调优

2026-10-28新增独立入口，按 cell、block、方向数依次调优，SVM 的 C 始终固定为1。每组使用全部6024张训练图像，比较现有324张测试图像的准确率；测试集参与参数选择，因此结果属于调参使用的测试集表现。

1. 固定方向数9、block 2×2，比较 cell 4×4、8×8、16×16像素。
2. 固定选定的 cell 和方向数9，比较 block 2×2、4×4个cell。
3. 固定选定的 cell、block，比较方向数9、6、12；方向数12由用户追加，18已取消。

每阶段取测试准确率最高者，并列时取特征维度较低者，再进入下一阶段；追加12后共6套不同配置。原始9/8/2配置复用已有C=1模型、完整特征并重放核验，其他配置重新提取特征和训练。新实验中的cell=4与block=4覆盖此前迁移参数的对应取值，但不将此前C=0.2的结果混入本轮比较。其余预处理和SVM设置沿用上节，固定1000次迭代，未收敛时中止并报错。

```bash
python tune_traffic_hog.py
# 独立复现需使用新的输出目录
python tune_traffic_hog.py --output outputs/traffic_hog_tuning_repeat
```

默认输出为`outputs/traffic_hog_tuning_20261028_v1/`。`protocol.json`保存搜索设置、图像路径与哈希和软件版本；每组保存模型、逐张预测、特征清单与结果；`trials.json`、`selection.json`和`summary.md`逐步记录结果、阶段选择及最终结论，完成后生成`verification.json`。同目录续跑检查协议、产物哈希和预测指标；不覆盖已完成实验。

特征缓存仅写入`mktemp -d`创建的系统临时目录，每组完成后删除，退出时清理整个临时目录。既有old/new实验文件保持原样。该入口与记录不修改课程报告；最终结论限于本轮搜索顺序、候选取值和当前数据划分。

当前已完成含方向数12补测在内的6组比较。cell阶段选4×4（94.14%，8×8与16×16均为93.52%）；固定cell=4×4时，block 4×4为95.06%，优于block 2×2的94.14%；固定cell=4×4、block=4×4、C=1时，方向数6、9、12分别为93.52%（303/324）、95.06%（308/324）、94.14%（305/324）。最终选择仍为方向数9、cell 4×4、block 4×4、C=1，测试准确率95.06%，比原始配置提高1.54个百分点、净多分对5张。6组训练准确率均为100%，均已收敛，并完成保存模型的全训练集和测试集预测重放核验。方向数12的SVM训练耗时591.11秒，方向数6为290.59秒；耗时受内存与换页影响，不作为严格的速度排名。

最初的方向数18候选训练未完成，暂停时已终止，现已按用户要求取消，不再运行、不参与选择；其记录保存在`interrupted_trials.json`及`protocol.json`的修订历史中。6组结果和结论见`summary.md`，本轮不修改课程报告。相同入口再次运行时只核验并复用全部完成结果，不启动新训练。

## HOG维度验证

本节验证要求3的指定参数，与要求2的7组参数对比独立。

实验要求第三点可单独执行，不运行 CIFAR-10 特征提取与分类：

```bash
python verify_hog_dimension.py
```

程序使用 `jinx.jpg` 的灰度图（432×768），设置方向数9、单元尺寸8×8像素、块尺寸16×16单元。实际返回的数组形状为 `(39,81,16,16,9)`，展开后的特征长度为7,278,336。程序检查维度推导、数值有限性，以及两种返回形式展开后的逐项一致性，并把参数、输出形状和环境版本写入 `outputs/hog_dimension/verification.json`。

## CIFAR-10 HOG 参数搜索

要求4当前只搜索 HOG 的 `orientations`、`pixels_per_cell` 和 `cells_per_block`。`tune_hog.py` 复用 `extractFeat.py` 导出的 `rgb2gray`、`hog` 和 `unpickle`，保持原有特征提取入口及已有 `.feat` 文件。搜索使用正方形 cell 和 block，命令中的 `o,p,b` 表示 `orientations=o, pixels_per_cell=(p,p), cells_per_block=(b,b)`。

- 固定灰度化系数与 `/255.0`、`L2-Hys` 归一化；不增加缩放、增强、颜色特征或特征标准化。
- 固定 `LinearSVC(C=1, max_iter=1000, tol=1e-4, dual="auto", random_state=20261022)`，与 `classifier.py --params old` 的设置一致。
- 从官方50000张训练图像中，按类别分层固定划分40000张训练、10000张验证。按用户确认的方案，后续所有组合均直接以40000张训练图像训练，不进行小样本筛选。
- 单参数比较、组合搜索与邻域细化均依据相同训练规模的验证准确率比较，记录训练准确率、特征维度、耗时和收敛状态。早期已完成的12次10000张训练实验保留为补充数据，不用于筛掉候选。
- 参数选定后，才用全部50000张训练图像分别训练基线和最佳候选，评估官方10000张测试图像。测试准确率不参与候选选择。

从本目录运行，例如：

```bash
python tune_hog.py --phase full --configs 9,8,2 9,4,2
python tune_hog.py --phase full --configs 6,4,2 12,4,2
python tune_hog.py --phase full --configs 8,4,2 10,4,2
python tune_hog.py --phase final
```

示例中的组合不表示已经完成的实验，实测范围以 `outputs/hog_tuning/trials.json` 为准。单参数阶段12组取值为：方向数4、6、9、12、18，cell边长4、6、8、10、16，block边长1、2、3、4；每次只改变一项。组合实验与细化均使用 `--phase full --configs ...`；已完成的同阶段组合会自动跳过。`--phase single/joint/refine` 保留早期10000张训练规模的复现功能，不用于当前主搜索。

`outputs/hog_tuning/` 保留完整实验记录：`protocol.json` 包含数据哈希和软件版本，`splits.npz` 包含划分索引，`extraction_parity.json` 包含原提取器的一致性检查，`trials.json` 包含各轮实测结果，`selection.json` 包含测试前锁定的参数，`final_evaluation.json` 包含基线和最佳候选的最终指标。最终模型、预测和最佳特征也保存在该目录，便于复核。`best_features.npy` 的前50000行是训练特征，后10000行是测试特征；对应标签在 `best_labels.npz`。

助手中间特征使用系统唯一临时目录，退出时清理；上述最终实验记录属于保留的实验产物。特征文件、模型、结果和数据仅保留在本地，不提交 Git。独立复现应使用新的 `--output` 目录；同目录续跑会检查提取器、数据、依赖版本和分类器设置是否一致。

参数定义可查阅 [scikit-image HOG 文档](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog) 和 [scikit-learn LinearSVC 文档](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)。原始 HOG 提取逻辑保留在 `extractFeat.py`。

使用独立 `--output` 子目录可同时训练不同组合。合并和绘图时运行 `python summarize_hog_tuning.py --merge-shards`；该程序读取主输出目录的 `shards/*/trials.json`，校验协议与划分完全一致后合并，拒绝冲突的重复记录。合并完成后在主目录执行 `--phase final`，自动选择所有40000张训练实验中的最佳候选。`summary.md`、`trials.csv` 和两张参数比较图由汇总程序生成；最终测试完成后再汇总一次即可更新。

记录的耗时为本机实际墙钟时间；同时运行的任务数会变化，因此这些耗时用于复现过程记录，不作为严格的参数速度排名。

## 验证次佳参数的完整训练集评估

在原40000张训练实验中，按验证准确率排名，第2名为 `6 / (4,4) / (4,4)`（54.73%），第3名为 `9 / (4,4) / (5,5)`（54.53%）。使用相同提取器和 SVM 设置，将这两组分别用全部50000张训练图像重训，评估官方10000张测试图像：

```bash
python evaluate_hog_runners_up.py --rank 2
python evaluate_hog_runners_up.py --rank 3
python summarize_hog_tuning.py
```

每组单独保存在 `outputs/hog_tuning/runners_up/`，包括协议、原验证排名、训练/测试准确率、分类别指标、混淆矩阵、模型与逐张预测。汇总时生成 `runners_up/comparison.json`，并将前三组的全量训练结果加入 `summary.md`。两组可以并行运行；训练过程中不运行汇总。

补充程序检查原数据哈希、提取器、依赖版本和分类器设置；保存模型与预测后重新核算指标并核对模型。原参数选择、首次最终评估、原模型和预测均以哈希确认保持一致。已完成的同一排名实验会直接复用；不保存额外的整套特征缓存。

## 固定 HOG 的 SVM 参数调优

```bash
python tune_svm.py --wait-for-hog
python plot_svm_tuning.py
```

程序等待两组次佳 HOG 补测完成，再从原验证前三组中，按50000张训练后的测试准确率选出 HOG 基线。参数、原模型、准确率和来源哈希锁定在 `outputs/svm_tuning/baseline.json`。本轮三组测试准确率依次为55.77%、55.31%、55.23%，因此固定 HOG 为方向数9、cell 4×4、block 4×4，基线分类器为 C=1、loss=squared_hinge。

按用户确定的方案，SVM 每组都直接使用全部50000张训练图像和官方另外10000张测试图像，不进行小样本筛选或40000张训练划分。官方测试集直接参与本轮参数选择，结果按“调参使用的测试集准确率”记录。HOG、图像处理、数据顺序、L2惩罚、容差和随机种子固定；同一份 HOG 特征贯穿所有实验。

loss固定为squared_hinge，比较 C 的数量级0.0001、0.001、0.01、0.1、1；若最优值位于边界，再按十倍扩展，最多扩展两次。在最佳区间按0.2、0.5、2、5倍细化，再比较相邻的根号2倍取值。C=1直接复用已完成并重放核验的50000张训练基线。按用户决定取消loss调优；此前启动的hinge训练已中止，没有完成结果，不参与选择。

本轮完成11个C取值，最佳为C=0.2、loss=squared_hinge，训练准确率62.94%、测试准确率56.35%，比固定HOG、C=1基线的55.77%提高0.58个百分点。最佳模型为`outputs/svm_tuning/models/C0.2_squared_hinge.joblib`。仅重建已完成结果的汇总和核验时使用`python tune_svm.py --finalize-only --jobs 1`，缺少完成记录会报错，不启动新的模型训练。

`dual="auto"` 在当前50000样本、3600维特征和squared_hinge设置下使用原始问题求解器；`max_iter` 从1000开始，仅在未收敛时增加至10000、50000，不以未收敛结果选择模型。所有模型均从头训练，没有小样本训练或 warm start。输出目录包括：`baseline.json`、`features.json`、`protocol.json`、`trials.json/csv`、`C_selection.json`、`selection.json`、`summary.md`、`verification.json`，以及逐组模型和预测。原始 HOG 记录、基线模型及其准确率保留。

同一阶段互不依赖的 C 取值默认用 `--jobs 2` 并行，阶段结束后统一选择，不改变每组50000张训练规模；可用 `--jobs 1` 顺序复现。已完成的逐组结果会复用，中断时未完成的点重新训练。`protocol.json` 的运行器历史保留执行方式变更和取消loss调优的记录；当前入口仅调C。

若选出的 HOG 与此前最佳组合相同，则只读复用 `outputs/hog_tuning/best_features.npy`；否则在 SVM 输出目录保存一份固定 HOG 特征，作为持续调参的工作文件。程序在训练前检查特征样本，并重放基线训练/测试预测，确认其准确率一致。

## HOG参数可视化与特征审阅

要求2使用独立入口，不运行CIFAR-10或交通标志分类，也不改动报告：

```bash
python compare_hog_parameters.py
# 独立复现时可指定新的空目录；非空目录会被拒绝，已有结果不被覆盖。
python compare_hog_parameters.py --output outputs/hog_parameter_review_repeat
```

同一张432×768的 `jinx.jpg` 按原 `visualize_hog.py` 的OpenCV灰度化方式处理，不缩放、不除以255、不增强。以方向数9、Cell 8×8像素、Block 2×2个Cell为基线，分别比较方向数4/9/18、Cell边长4/8/16、Block边长1/2/4；扣除重复基线，共7组不同配置。固定 `L2-Hys`、`transform_sqrt=False`。

2026-10-28的首轮结果位于本地 `outputs/hog_parameter_review_20261028_v1/`，默认的新运行使用独立时间戳目录。审阅入口为 `review.html`，另有三张参数比较PNG、完整特征数值分布图、`summary.csv`、`results.json`、`verification.json` 和文件哈希清单 `files.json`。本轮状态为“实验结果待共同审阅，尚未写入报告”。

每组保存公开HOG返回的完整一维 `features.npy`、原始 `hog_vis.npy`、显示用 `hog_vis.png` 和Block归一化前的 `cell_histograms.npy`。按 `metadata.json` 中的 `block_shape` 重排一维特征即可恢复五维Block结构，不重复保存整套数组。`target_cell.npz` 保存固定目标Cell的原始直方图、指定Block中的归一化向量，以及该Cell在所有覆盖Block中的归一化向量及Block索引。`orientation_histograms.csv` 保存目标Cell及全图平均的方向统计；`gradients.npz` 保存行列梯度、幅值和无符号方向。

公开HOG接口不直接返回Cell原始直方图，脚本调用安装版本的私有梯度和Cython直方图函数恢复中间值；逐组核验中间值能够精确重建公开函数返回的全部Block特征与原始可视化。该路径依赖具体实现，环境升级后需重新通过检查；依赖版本、输入及实现哈希记录在 `provenance.json`。程序还核验维度公式、两种 `feature_vector` 返回形式、数值有限性和保存后数组一致性。

审阅图采用统一平方根显示变换和全部7组共用的亮度范围，99.5百分位截断仅用于显示，原始数组不变。固定局部区域为行32:160、列352:480，目标定位像素为(y=80,x=432)；Cell大小不同时目标Cell覆盖范围不同。Block组使用同一Cell，选取以该Cell为左上角的Block，比较其中该Cell的同维方向向量。不同参数完整向量的维度或元素语义不同，不直接逐项相减；图像外观与数值分布也不作为分类优劣的证据。
