"""用法：python geometry_all_metric.py out_metric280 results_metric_518

直接评估米制深度，不拟合尺度、不取倒数、不裁剪预测值。
使用 nyu_data/reference 中的 valid 掩码，与原 geometry_all.py 相同。
输入尺寸、图片清单从各目录的 inference_timing.json 读取，避免重复处理派生文件。
在预测目录导出 *_depth_m.npy、*_pred.ply 和 *_gt.ply。
逐图指标和逐图等权均值分别写入 evaluation/metric_evaluation_metrics.csv
和 evaluation/metric_evaluation_means.csv。RMSE 均值是各图 RMSE 的算术平均，
不是将所有像素合并后计算的 RMSE。相对深度的旧评估使用真值对齐，口径不同。
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parent
PROTOCOL = 'raw_metric_no_alignment_no_clipping'
METRICS = ('abs_rel', 'depth_rmse_m', 'xyz_rmse_m')


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
    sid, out = path.stem, path.parent
    z = np.load(path).astype(float)
    with np.load(ROOT / 'nyu_data' / 'reference' / f'{sid}.npz') as data:
        g, K, valid = (data[key] for key in ('depth_m', 'K', 'valid'))
    values, P, G = evaluate_depth(z, g, K, valid)

    rgb_path = ROOT / 'nyu_data' / 'rgb' / f'{sid}.png'
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
        'prediction_dir': str(out),
        'image_id': sid,
        'valid_pixels': int(valid.sum()),
        'depth_processing': PROTOCOL,
        **values,
    }
    print(
        f'[{input_size}] {sid}: AbsRel = {values["abs_rel"]:.6f}, '
        f'Depth RMSE (m) = {values["depth_rmse_m"]:.6f}, '
        f'XYZ RMSE (m) = {values["xyz_rmse_m"]:.6f}'
    )
    return row


def collect_jobs(folders):
    jobs = []
    seen = set()
    for folder in folders:
        resolved = folder.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        with (folder / 'inference_timing.json').open(encoding='utf-8') as handle:
            metadata = json.load(handle)
        if metadata.get('depth_unit') != 'meters':
            raise ValueError(f'{folder} 缺少米制深度标记，请使用 Metric 推理输出')
        size = metadata['input_size']
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f'{folder} 的 input_size 无效')
        ids = [Path(item['filename']).stem for item in metadata['images']]
        if not ids or len(set(ids)) != len(ids) or metadata['image_count'] != len(ids):
            raise ValueError(f'{folder} 的图片清单为空、重复或与 image_count 不符')
        for sid in sorted(ids):
            path = folder / f'{sid}.npy'
            for required in (path, ROOT / 'nyu_data' / 'reference' / f'{sid}.npz',
                             ROOT / 'nyu_data' / 'rgb' / f'{sid}.png'):
                if not required.is_file():
                    raise FileNotFoundError(required)
            jobs.append((path, size))
    return jobs


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['input_size'], row['prediction_dir']), []).append(row)
    return [
        {
            'input_size': size,
            'prediction_dir': folder,
            'image_count': len(group),
            'averaging_method': 'equal_weight_per_image',
            'depth_processing': PROTOCOL,
            **{f'{key}_mean': float(np.mean([row[key] for row in group])) for key in METRICS},
        }
        for (size, folder), group in sorted(groups.items())
    ]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+', type=Path, help='Metric 推理结果目录')
    args = parser.parse_args()
    try:
        jobs = collect_jobs(args.folders)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    rows = [process_prediction(path, size) for path, size in jobs]
    means = summarize(rows)
    detail_path = ROOT / 'evaluation' / 'metric_evaluation_metrics.csv'
    mean_path = ROOT / 'evaluation' / 'metric_evaluation_means.csv'
    write_csv(detail_path, rows)
    write_csv(mean_path, means)
    print(f'已处理 {len(rows)} 张预测图，逐图指标：{detail_path}')
    print(f'逐图等权均值：{mean_path}')
    for row in means:
        print(f'[{row["input_size"]}] {row["image_count"]} 张图均值：'
              + ', '.join(f'{key} = {row[key + "_mean"]:.6f}' for key in METRICS))


if __name__ == '__main__':
    main()
