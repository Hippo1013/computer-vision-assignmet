"""助教 3DCNNs 权重的 TensorFlow 2 推理适配。

网络连接、SAME 填充、8 通道一组的 GN、无 ReLU 的 2D 反卷积、
3D BN 的当前样本统计及半像素单应变换均沿用 YoYo000/MVSNet。
使用原始 checkpoint 张量，不重新训练，不换用其他预训练模型。
"""

# Network topology adapted from Copyright 2019 Yao Yao, HKUST (MIT).
# License: ../MVSNet/LICENSE
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf


class MVSNet:
    def __init__(self, checkpoint, threads=6):
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        reader = tf.train.load_checkpoint(str(checkpoint))
        self.weights = {
            k: tf.constant(reader.get_tensor(k))
            for k in reader.get_variable_to_shape_map()
            if ("/kernel" in k or "/gn/" in k or "/bn/" in k) and "/Adam" not in k
        }
        self.used = set()

    def weight(self, key):
        self.used.add(key)
        return self.weights[key]

    def conv(self, x, name, stride=1, norm="gn", transpose=False):
        kernel = self.weight(name + "/kernel")
        rank = len(x.shape) - 2
        if transpose:
            shape = tf.concat([tf.shape(x)[:1], tf.shape(x)[1:-1] * stride, [kernel.shape[-2]]], 0)
            op = tf.nn.conv2d_transpose if rank == 2 else tf.nn.conv3d_transpose
            y = op(x, kernel, shape, strides=[1] + [stride] * rank + [1], padding="SAME")
        else:
            op = tf.nn.conv2d if rank == 2 else tf.nn.conv3d
            y = op(x, kernel, strides=[1] + [stride] * rank + [1], padding="SAME")
        if norm == "gn":
            shape = tf.shape(y)
            channels = y.shape[-1]
            groups = max(1, channels // 8)
            z = tf.transpose(y, [0, 3, 1, 2])
            z = tf.reshape(z, [shape[0], groups, channels // groups, shape[1], shape[2]])
            mean, var = tf.nn.moments(z, [2, 3, 4], keepdims=True)
            z = (z - mean) * tf.math.rsqrt(var + 1e-5)
            z = tf.reshape(z, [shape[0], channels, shape[1], shape[2]])
            y = tf.transpose(z, [0, 2, 3, 1]) * self.weight(name + "/gn/gamma") + self.weight(
                name + "/gn/beta"
            )
        elif norm == "bn":
            # 原始 test.py 以 is_training=True 建图；不用 moving_mean/variance。
            mean, var = tf.nn.moments(y, list(range(rank + 1)))
            y = tf.nn.batch_normalization(
                y, mean, var, self.weight(name + "/bn/beta"), self.weight(name + "/bn/gamma"), 1e-5
            )
        return tf.nn.relu(y) if norm and not (transpose and rank == 2) else y

    @tf.function(reduce_retracing=True)
    def features(self, image):
        down = {0: image}
        for i in range(1, 5):
            down[i] = self.conv(down[i - 1], f"2dconv{i}_0", 2)
        skip = {}
        for i in range(5):
            skip[i] = self.conv(self.conv(down[i], f"2dconv{i}_1"), f"2dconv{i}_2")
        x = skip[4]
        for i in range(5, 9):
            x = self.conv(x, f"2dconv{i}_0", 2, transpose=True)
            x = tf.concat([x, skip[8 - i]], -1)
            x = self.conv(self.conv(x, f"2dconv{i}_1"), f"2dconv{i}_2")
        for i in (9, 10):
            x = self.conv(x, f"conv{i}_0", 2)
            x = self.conv(x, f"conv{i}_1")
            x = self.conv(x, f"conv{i}_2", norm="gn" if i == 9 else None)
        return x

    @tf.function(reduce_retracing=True)
    def regularize(self, cost):
        down = {0: cost}
        for i in range(1, 4):
            down[i] = self.conv(down[i - 1], f"3dconv{i}_0", 2, norm="bn")
        skip = {i: self.conv(down[i], f"3dconv{i}_1", norm="bn") for i in range(4)}
        x = skip[3]
        for i in range(4, 7):
            x = self.conv(x, f"3dconv{i}_0", 2, norm="bn", transpose=True) + skip[6 - i]
        return self.conv(x, "3dconv6_2", norm=None)[0, ..., 0]

    def predict(self, features, cameras, depths):
        """逐深度片构建代价体，减少临时内存；输出相机 Z 深度和四格概率和。"""
        ref = features[0]
        Kref, Eref = cameras[0]
        h, w = ref.shape[1:3]
        homographies = []
        for K, E in cameras[1:]:
            Rref = Eref[:3, :3]
            R = E[:3, :3]
            cref = -Rref.T @ Eref[:3, 3:4]
            csrc = -R.T @ E[:3, 3:4]
            middle = (
                np.eye(3, dtype=np.float32)[None]
                - (csrc - cref)[None] @ Rref[None, 2:3, :] / depths[:, None, None]
            )
            H = K[None] @ (R[None] @ (middle @ (Rref.T @ np.linalg.inv(Kref))[None]))
            # 坐标从像素中心 (u+.5,v+.5) 转到 TF transform 的整数索引。
            shift = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], np.float32)
            H = np.linalg.inv(shift)[None] @ H @ shift[None]
            H = H / H[:, 2:3, 2:3]
            homographies.append(H.reshape(-1, 9)[:, :8].astype(np.float32))
        costs = []
        # 小批深度并行变换，避免重复特征提取及过大的 N*D*H*W*C 临时数组。
        for start in range(0, len(depths), 8):
            n = min(8, len(depths) - start)
            total = tf.repeat(ref, n, axis=0)
            square = total * total
            for feat, transforms in zip(features[1:], homographies):
                warped = tf.raw_ops.ImageProjectiveTransformV3(
                    images=tf.repeat(feat, n, axis=0),
                    transforms=transforms[start : start + n],
                    output_shape=[h, w],
                    interpolation="BILINEAR",
                    fill_mode="CONSTANT",
                    fill_value=0.0,
                )
                total = total + warped
                square = square + warped * warped
            costs.append(square / len(features) - tf.square(total / len(features)))
        cost = tf.concat(costs, axis=0)[None]
        del costs, total, square
        probability = tf.nn.softmax(-self.regularize(cost), axis=0)
        depth = tf.reduce_sum(probability * depths[:, None, None], axis=0)
        index = (depth - depths[0]) / (depths[1] - depths[0])
        left = tf.cast(tf.floor(index), tf.int32)
        right = tf.cast(tf.math.ceil(index), tf.int32)
        yy, xx = tf.meshgrid(tf.range(h), tf.range(w), indexing="ij")
        confidence = tf.zeros_like(depth)
        # 沿用上游四项求和：整数索引或边界可能重复计数，不强制归一化。
        for k in (left, left - 1, right, right + 1):
            confidence += tf.gather_nd(
                probability, tf.stack([tf.clip_by_value(k, 0, len(depths) - 1), yy, xx], -1)
            )
        return depth.numpy(), confidence.numpy()
