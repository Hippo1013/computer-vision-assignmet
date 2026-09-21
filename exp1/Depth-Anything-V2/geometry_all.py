"""用法：python geometry_all.py out280 results_518

逐图对齐、评估并导出点云，三个误差指标汇总到 evaluation/evaluation_metrics.csv。
拟合和评估使用同一批有效像素，结果为利用真值对齐后的误差。
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parent
INPUT_SIZES = {'out280': 280, 'results_518': 518}


def process_prediction(path, input_size):
    sid, out = path.stem, path.parent
    q = np.load(path).astype(float)
    with np.load(ROOT / 'nyu_data' / 'reference' / f'{sid}.npz') as data:
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
        'prediction_dir': str(out),
        'image_id': sid,
        'valid_pixels': int(valid.sum()),
        'aligned_abs_rel': float(np.mean(np.abs(e) / g[valid])),
        'depth_rmse_m': float(np.sqrt(np.mean(e ** 2))),
        'xyz_rmse_m': float(np.sqrt(np.mean(xyz2))),
    }

    rgb_path = ROOT / 'nyu_data' / 'rgb' / f'{sid}.png'
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
    parser.add_argument('folders', nargs='+', type=Path, help='out280 和 results_518 的路径')
    args = parser.parse_args()

    jobs = []
    for folder in dict.fromkeys(args.folders):
        if not folder.is_dir() or folder.name not in INPUT_SIZES:
            parser.error(f'请提供存在的 out280 或 results_518 目录：{folder}')
        # 跳过本脚本生成的深度图，保证再次运行时不会重复处理。
        predictions = sorted(
            path for path in folder.glob('*.npy')
            if not path.stem.endswith('_depth_m')
        )
        if not predictions:
            parser.error(f'{folder} 中没有预测 .npy 文件')
        jobs.extend((path, INPUT_SIZES[folder.name]) for path in predictions)

    rows = [process_prediction(path, size) for path, size in jobs]
    csv_path = ROOT / 'evaluation' / 'evaluation_metrics.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'已处理 {len(rows)} 张预测图，评估表格：{csv_path}')


if __name__ == '__main__':
    main()
