#!/usr/bin/env python3
"""用已知平面检查反投影、相机变换、遮挡边界及深度过滤。"""
import numpy as np
from mvsnet_local.geometry import backproject, pixel_grid, reproject
from mvsnet_local.io import ROOT, save_json


def main():
    K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]], np.float32)
    E = np.eye(4, dtype=np.float32)
    depth = np.full((48, 64), 600, np.float32)
    xyz = backproject(depth, K, E)
    projected = K @ xyz.T
    uv = projected[:2] / projected[2]
    roundtrip = float(np.max(np.abs(uv - pixel_grid(depth.shape)[:2])))
    assert roundtrip < 1e-4
    source_E = E.copy()
    source_E[0, 3] = -30
    ref = {"depth": depth, "K": K, "E": E}
    src = {"depth": depth, "K": K, "E": source_E}
    mask, _, err, rel, valid = reproject(ref, src)
    assert mask[2:-2, 8:-2].all() and err[valid].max() < 1e-3 and rel[valid].max() < 1e-5
    assert not mask[:, :5].any()
    wrong = {**src, "depth": depth * 1.1}
    bad, *_ = reproject(ref, wrong)
    assert not bad.any()
    result = {
        "backprojection_roundtrip_max_px": roundtrip,
        "translated_plane_max_px": float(err[valid].max()),
        "outside_image_rejected": True,
        "ten_percent_wrong_depth_rejected": True,
        "passed": True,
    }
    out = ROOT / "outputs/summary"
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "geometry_checks.json", result)
    print(result)


if __name__ == "__main__":
    main()
