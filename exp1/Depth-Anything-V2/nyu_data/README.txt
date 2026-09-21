NYU Depth V2 单目深度实验数据（仅数据）

解压后将 nyu_lab 放在学生自行下载的 Depth-Anything-V2 仓库根目录。
rgb/ 中仅包含 8 张 640×480 的 RGB PNG，可直接交给官方 run.py。
reference/ 中同名 NPZ 包含：
  depth_m：已配准的 Kinect 实测深度 rawDepths，单位米，保留原始缺失值。
  K：相机内参，0 起始像素坐标。fx=518.8579011745019，fy=519.4696111212749。
     cx=324.58244941119034，cy=252.73616633400465。
  valid：有效像素掩膜。同一张图的全部有效像素用于求一组 a、b，再评估对齐后的误差。

数据来源：https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
原始文件：nyu_depth_v2_labeled.mat，约 2.97 GB。
MATLAB 样本编号：1、101、201、401、601、801、1001、1201。
内参来自原 NYUv2 toolbox camera_params.m，主点坐标由 MATLAB 的 1 起始转换为 0 起始。
代码镜像：https://github.com/jjhartmann/Toolbox-NYU-Depth-V2/blob/master/camera_params.m

处理说明：只提取图像/实测深度、转换存储格式、加入内参和固定掩膜。
不修改原实测深度值，不使用填补后的 depths 作为实测真值。
有效域：有限且 0.1 < Z < 10 米，NumPy 切片 [45:471,41:601] 的区域。
不划分标定/评估子集。详细来源、数量和校验值见 metadata.json。
评估名称为“逐图尺度与偏移对齐后的误差”，结果利用了真实深度的尺度信息。
它不代表模型独立预测绝对距离的能力，也不是官方完整基准测试的复现。
这是教学样例，不是官方完整测试划分。Kinect 测量也存在噪声。

本包不含任何模型代码、网络权重、学生程序、预测结果或参考答案。
仅用于教学/研究；使用原数据时请遵循发布方说明并引用 NYU Depth V2。
