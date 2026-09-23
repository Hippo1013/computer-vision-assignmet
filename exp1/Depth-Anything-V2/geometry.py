"""单图相对深度评估：python geometry.py nyu_0001 --input-size 518。"""

import argparse

import cv2
import numpy as np
import open3d as o3d


from experiment_paths import REFERENCE, RGB, SIZES, prediction_dir, result_dir
from summarize_results import save_group, rebuild_summary

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('image_id')
parser.add_argument('--input-size', type=int, choices=SIZES, default=518)
args = parser.parse_args()
sid = args.image_id
if not sid.startswith('nyu_') or not sid[4:].isdigit():
    parser.error('图片编号格式应为 nyu_0001')
source = prediction_dir('relative', args.input_size)
out = result_dir('relative', args.input_size, 'aligned')
out.mkdir(parents=True, exist_ok=True)

q = np.load(source / (sid + '.npy')).astype(float)
d = np.load(REFERENCE / (sid + '.npz'))
g, K, valid = (d[k] for k in ['depth_m', 'K', 'valid'])
assert q.shape == g.shape and np.isfinite(q).all()

A = np.c_[q[valid], np.ones(valid.sum())]
a, b = np.linalg.lstsq(A, 1 / g[valid], rcond=None)[0]
assert a > 0 and np.isfinite(a + b)

inv = a * q + b
z = np.divide(1, inv, out=np.full_like(inv, 10), where=inv > 0)
z = np.clip(z, 0.1, 10)
np.save(out / (sid + '_depth_m.npy'), z)

v, u = np.indices(z.shape)
rays = np.stack([u, v, np.ones_like(u)], axis=-1)
rays = rays @ np.linalg.inv(K).T
P, G = z[..., None] * rays, g[..., None] * rays

e = z[valid] - g[valid]
print(sid, 'Aligned AbsRel =', np.mean(np.abs(e) / g[valid]))
print('Depth RMSE (m) =', np.sqrt(np.mean(e ** 2)))

xyz2 = np.sum((P[valid] - G[valid]) ** 2, axis=1)
print('XYZ RMSE (m) =', np.sqrt(np.mean(xyz2)))

rgb = cv2.imread(str(RGB / (sid + '.png')))[..., ::-1] / 255.0

for name, points in [('pred', P), ('gt', G)]:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[valid])
    cloud.colors = o3d.utility.Vector3dVector(rgb[valid])
    target = out / (sid + '_' + name + '.ply')
    if not o3d.io.write_point_cloud(str(target), cloud):
        raise OSError(f'点云保存失败：{target}')

save_group([{
    'image_id': sid, 'valid_pixels': int(valid.sum()),
    'alignment_a': float(a), 'alignment_b': float(b),
    'abs_rel': float(np.mean(np.abs(e) / g[valid])),
    'depth_rmse_m': float(np.sqrt(np.mean(e ** 2))),
    'xyz_rmse_m': float(np.sqrt(np.mean(xyz2))),
}], 'relative', args.input_size, 'aligned', merge=True)
rebuild_summary()
