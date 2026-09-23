"""用法：python geometry_metric_aligned.py --input-size 280 518

先对米制预测取倒数，再按相对版规则对齐、裁剪和评估。
读取 outputs/metric/<尺寸>/prediction，写入同级 aligned，并自动刷新汇总表。
"""

import argparse

import cv2
import numpy as np
import open3d as o3d


from experiment_paths import REFERENCE, RGB, SIZES, ALIGNED_PROTOCOL, collect_predictions, result_dir
from summarize_results import save_group, rebuild_summary

METRICS = ('aligned_abs_rel', 'depth_rmse_m', 'xyz_rmse_m')


def align_metric_depth(d, g, valid):
    """先 q=1/d，再拟合 a*q+b ≈ 1/g；与相对版采用相同的对齐规则。"""
    d = np.asarray(d, dtype=float)
    if d.ndim != 2 or d.shape != g.shape or valid.shape != g.shape:
        raise ValueError('预测、真值和 valid 必须是尺寸相同的二维数组')
    if valid.dtype != np.bool_ or not valid.any():
        raise ValueError('valid 必须是非空的布尔掩膜')
    if not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError('Metric 预测必须全部为有限正数，才能取倒数')
    if not np.isfinite(g[valid]).all() or np.any(g[valid] <= 0):
        raise ValueError('有效像素上的真实深度必须为有限正数')

    q = 1.0 / d
    if not np.isfinite(q).all():
        raise ValueError('Metric 预测的倒数出现非有限值')
    A = np.c_[q[valid], np.ones(valid.sum())]
    a, b = np.linalg.lstsq(A, 1 / g[valid], rcond=None)[0]
    if a <= 0 or not np.isfinite(a + b):
        raise ValueError(f'逆深度对齐失败：a={a}, b={b}')
    inv = a * q + b
    z = np.divide(1, inv, out=np.full_like(inv, 10), where=inv > 0)
    return np.clip(z, 0.1, 10), float(a), float(b)


def process_prediction(path, input_size):
    sid = path.stem
    with np.load(REFERENCE / f'{sid}.npz') as data:
        g, K, valid = (data[key] for key in ('depth_m', 'K', 'valid'))
    z, a, b = align_metric_depth(np.load(path), g, valid)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError(f'{sid} 的内参必须是有限的 3x3 矩阵')
    v, u = np.indices(z.shape)
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(K).T
    P, G = z[..., None] * rays, g[..., None] * rays
    e = z[valid] - g[valid]
    values = {
        'aligned_abs_rel': float(np.mean(np.abs(e) / g[valid])),
        'depth_rmse_m': float(np.sqrt(np.mean(e ** 2))),
        'xyz_rmse_m': float(np.sqrt(np.mean(np.sum((P[valid] - G[valid]) ** 2, axis=1)))),
    }
    bgr = cv2.imread(str(RGB / f'{sid}.png'))
    if bgr is None or bgr.shape[:2] != z.shape:
        raise ValueError(f'{sid} 的 RGB 无法读取或尺寸不匹配')
    rgb = bgr[..., ::-1] / 255.0
    out = result_dir('metric', input_size, 'aligned')
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f'{sid}_depth_m.npy', z)
    for name, points in [('pred', P), ('gt', G)]:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[valid])
        cloud.colors = o3d.utility.Vector3dVector(rgb[valid])
        target = out / f'{sid}_{name}.ply'
        if not o3d.io.write_point_cloud(str(target), cloud):
            raise OSError(f'点云保存失败：{target}')
    print(f'[{input_size}] {sid}: ' + ', '.join(f'{k} = {values[k]:.6f}' for k in METRICS))
    return {
        'input_size': input_size, 'prediction_dir': str(path.parent),
        'output_dir': str(out), 'image_id': sid, 'valid_pixels': int(valid.sum()),
        'depth_processing': ALIGNED_PROTOCOL, 'alignment_a': a, 'alignment_b': b, **values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-size', nargs='+', type=int, choices=SIZES, default=list(SIZES))
    args = parser.parse_args()
    try:
        jobs = collect_predictions('metric', args.input_size)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    rows = [process_prediction(path, size) for path, size in jobs]
    for size in dict.fromkeys(args.input_size):
        save_group([row for row in rows if row['input_size'] == size], 'metric', size, 'aligned')
    rebuild_summary()


if __name__ == '__main__':
    main()
