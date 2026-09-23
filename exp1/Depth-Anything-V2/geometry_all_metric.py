"""用法：python geometry_all_metric.py --input-size 280 518

直接评估原始米制深度，不对齐、不裁剪。
读取 outputs/metric/<尺寸>/prediction，写入同级 raw，并自动刷新汇总表。
"""

import argparse

import cv2
import numpy as np
import open3d as o3d


from experiment_paths import REFERENCE, RGB, SIZES, RAW_PROTOCOL, collect_predictions, result_dir
from summarize_results import save_group, rebuild_summary


def evaluate_depth(z, g, K, valid):
    """在真值有效像素上评估原始米制深度，并返回两组点云坐标。"""
    if z.shape != g.shape or valid.shape != g.shape or z.ndim != 2:
        raise ValueError('预测深度、真值和有效掩码的尺寸必须一致且为二维')
    if valid.dtype != np.bool_ or not valid.any():
        raise ValueError('valid 必须是非空的布尔掩码')
    if not np.isfinite(z).all() or np.any(z <= 0):
        raise ValueError('预测深度必须全部为有限正数，不能静默排除错误预测')
    if not np.isfinite(g[valid]).all() or np.any(g[valid] <= 0):
        raise ValueError('有效像素上的真实深度必须为有限正数')
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError('相机内参必须是有限的 3x3 矩阵')

    v, u = np.indices(z.shape)
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(K).T
    P, G = z[..., None] * rays, g[..., None] * rays
    e = z[valid] - g[valid]
    xyz2 = np.sum((P[valid] - G[valid]) ** 2, axis=1)
    metrics = {
        'abs_rel': float(np.mean(np.abs(e) / g[valid])),
        'depth_rmse_m': float(np.sqrt(np.mean(e ** 2))),
        'xyz_rmse_m': float(np.sqrt(np.mean(xyz2))),
    }
    return metrics, P, G


def process_prediction(path, input_size):
    sid = path.stem
    out = result_dir('metric', input_size, 'raw')
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(path).astype(float)
    with np.load(REFERENCE / f'{sid}.npz') as data:
        g, K, valid = (data[key] for key in ('depth_m', 'K', 'valid'))
    values, P, G = evaluate_depth(z, g, K, valid)

    rgb_path = RGB / f'{sid}.png'
    bgr = cv2.imread(str(rgb_path))
    if bgr is None or bgr.shape[:2] != z.shape:
        raise ValueError(f'无法读取图片或图片尺寸不匹配：{rgb_path}')
    rgb = bgr[..., ::-1] / 255.0
    # 保留原 geometry_all.py 的派生文件命名，内容仍为原始米制预测。
    np.save(out / f'{sid}_depth_m.npy', z)
    for name, points in [('pred', P), ('gt', G)]:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[valid])
        cloud.colors = o3d.utility.Vector3dVector(rgb[valid])
        cloud_path = out / f'{sid}_{name}.ply'
        if not o3d.io.write_point_cloud(str(cloud_path), cloud):
            raise OSError(f'点云保存失败：{cloud_path}')

    row = {
        'input_size': input_size,
        'prediction_dir': str(path.parent),
        'image_id': sid,
        'valid_pixels': int(valid.sum()),
        'depth_processing': RAW_PROTOCOL,
        **values,
    }
    print(
        f'[{input_size}] {sid}: AbsRel = {values["abs_rel"]:.6f}, '
        f'Depth RMSE (m) = {values["depth_rmse_m"]:.6f}, '
        f'XYZ RMSE (m) = {values["xyz_rmse_m"]:.6f}'
    )
    return row


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
        save_group([row for row in rows if row['input_size'] == size], 'metric', size, 'raw')
    rebuild_summary()


if __name__ == '__main__':
    main()
