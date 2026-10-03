"""相机 Z 深度反投影、跨视角往返检查与颜色点云融合。"""

import cv2
import numpy as np


def pixel_grid(shape):
    y, x = np.indices(shape, dtype=np.float32)
    return np.stack([x + 0.5, y + 0.5, np.ones_like(x)], 0).reshape(3, -1)


def backproject(depth, K, E):
    camera = (np.linalg.inv(K) @ pixel_grid(depth.shape)) * depth.reshape(1, -1)
    return (np.linalg.inv(E)[:3, :3] @ camera + np.linalg.inv(E)[:3, 3:4]).T


def reproject(ref, source, pixel_threshold=1.0, depth_threshold=0.01):
    d = ref["depth"]
    ds = source["depth"]
    h, w = ds.shape
    xyz = backproject(d, ref["K"], ref["E"]).T
    X = source["E"][:3, :3] @ xyz + source["E"][:3, 3:4]
    projected = source["K"] @ X
    denom = np.where(np.abs(projected[2]) > 1e-8, projected[2], np.nan)
    u = (projected[0] / denom - 0.5).reshape(d.shape)
    v = (projected[1] / denom - 0.5).reshape(d.shape)
    inside = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < w - 1)
        & (v >= 0)
        & (v < h - 1)
        & (X[2].reshape(d.shape) > 0)
    )
    us = np.nan_to_num(u, nan=-1).astype(np.float32)
    vs = np.nan_to_num(v, nan=-1).astype(np.float32)
    sampled = cv2.remap(ds, us, vs, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    source_rays = np.linalg.inv(source["K"]) @ np.stack(
        [us + 0.5, vs + 0.5, np.ones_like(us)], 0
    ).reshape(3, -1)
    source_xyz = source_rays * sampled.reshape(1, -1)
    relative = ref["E"] @ np.linalg.inv(source["E"])
    returned = relative[:3, :3] @ source_xyz + relative[:3, 3:4]
    reproj = ref["K"] @ returned
    denominator = np.where(np.abs(reproj[2]) > 1e-8, reproj[2], np.nan)
    uv = reproj[:2] / denominator
    refpixels = pixel_grid(d.shape)[:2]
    pixel_error = np.linalg.norm(uv - refpixels, axis=0).reshape(d.shape)
    returned_depth = returned[2].reshape(d.shape)
    relative_error = np.abs(returned_depth - d) / np.maximum(d, 1e-8)
    valid = inside & (sampled > 0) & (returned_depth > 0) & np.isfinite(pixel_error) & (d > 0)
    consistent = valid & (pixel_error < pixel_threshold) & (relative_error < depth_threshold)
    return consistent, returned_depth, pixel_error, relative_error, valid


def voxel_average(xyz, rgb, size=0.5):
    if size <= 0:
        return xyz, rgb
    keys = np.floor(xyz / size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = np.bincount(inverse)
    position = np.column_stack([np.bincount(inverse, weights=xyz[:, i]) / count for i in range(3)])
    color = np.column_stack([np.bincount(inverse, weights=rgb[:, i]) / count for i in range(3)])
    return position.astype(np.float32), np.rint(color).clip(0, 255).astype(np.uint8)
