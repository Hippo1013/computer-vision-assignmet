import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


sid, folder = sys.argv[1:3]
out = Path(folder)

q = np.load(out / (sid + '.npy')).astype(float)
d = np.load('nyu_data/reference/' + sid + '.npz')
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

rgb = cv2.imread('nyu_data/rgb/' + sid + '.png')[..., ::-1] / 255.0

for name, points in [('pred', P), ('gt', G)]:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[valid])
    cloud.colors = o3d.utility.Vector3dVector(rgb[valid])
    o3d.io.write_point_cloud(str(out / (sid + '_' + name + '.ply')), cloud)
