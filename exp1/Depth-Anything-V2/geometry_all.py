"""用法：python geometry_all.py --input-size 280 518

读取 outputs/relative/<尺寸>/prediction，写入同级 aligned，并自动刷新汇总表。
拟合和评估使用同一批有效像素，结果为利用真值对齐后的误差。
"""

import argparse

import cv2
import numpy as np
import open3d as o3d


from experiment_paths import REFERENCE, RGB, SIZES, collect_predictions, result_dir
from summarize_results import save_group, rebuild_summary


def process_prediction(path, input_size):
    sid = path.stem
    out = result_dir('relative', input_size, 'aligned')
    out.mkdir(parents=True, exist_ok=True)
    q = np.load(path).astype(float)
    with np.load(REFERENCE / f'{sid}.npz') as data:
        g, K, valid = (data[key] for key in ['depth_m', 'K', 'valid'])
    assert q.shape == g.shape and np.isfinite(q).all(), path

    # 用全部有效像素拟合当前图片共用的一组 a、b。
    A = np.c_[q[valid], np.ones(valid.sum())]
    a, b = np.linalg.lstsq(A, 1 / g[valid], rcond=None)[0]
    assert a > 0 and np.isfinite(a + b), path

    inv = a * q + b
    z = np.divide(1, inv, out=np.full_like(inv, 10), where=inv > 0)
    z = np.clip(z, 0.1, 10)
    np.save(out / f'{sid}_depth_m.npy', z)

    # 将预测深度和真实深度转换为三维点。
    v, u = np.indices(z.shape)
    rays = np.stack([u, v, np.ones_like(u)], axis=-1)
    rays = rays @ np.linalg.inv(K).T
    P, G = z[..., None] * rays, g[..., None] * rays

    e = z[valid] - g[valid]
    xyz2 = np.sum((P[valid] - G[valid]) ** 2, axis=1)
    metrics = {
        'input_size': input_size,
        'prediction_dir': str(path.parent),
        'image_id': sid,
        'valid_pixels': int(valid.sum()),
        'alignment_a': float(a),
        'alignment_b': float(b),
        'aligned_abs_rel': float(np.mean(np.abs(e) / g[valid])),
        'depth_rmse_m': float(np.sqrt(np.mean(e ** 2))),
        'xyz_rmse_m': float(np.sqrt(np.mean(xyz2))),
    }

    rgb_path = RGB / f'{sid}.png'
    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise FileNotFoundError(f'无法读取图片：{rgb_path}')
    rgb = bgr[..., ::-1] / 255.0

    for name, points in [('pred', P), ('gt', G)]:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[valid])
        cloud.colors = o3d.utility.Vector3dVector(rgb[valid])
        cloud_path = out / f'{sid}_{name}.ply'
        if not o3d.io.write_point_cloud(str(cloud_path), cloud):
            raise OSError(f'点云保存失败：{cloud_path}')

    print(
        f'[{input_size}] {sid}: '
        f'Aligned AbsRel = {metrics["aligned_abs_rel"]:.6f}, '
        f'Depth RMSE (m) = {metrics["depth_rmse_m"]:.6f}, '
        f'XYZ RMSE (m) = {metrics["xyz_rmse_m"]:.6f}'
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-size', nargs='+', type=int, choices=SIZES, default=list(SIZES))
    args = parser.parse_args()

    try:
        jobs = collect_predictions('relative', args.input_size)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    rows = [process_prediction(path, size) for path, size in jobs]
    for size in dict.fromkeys(args.input_size):
        save_group([row for row in rows if row['input_size'] == size], 'relative', size, 'aligned')
    rebuild_summary()


if __name__ == '__main__':
    main()
