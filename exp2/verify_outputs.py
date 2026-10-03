#!/usr/bin/env python3
"""逐项核对正式输出、相机坐标、融合掩膜和汇总数值。"""
import csv, json, hashlib
from pathlib import Path
import numpy as np
from mvsnet_local.io import ROOT, CHECKPOINT, read_pairs, read_pfm, save_json
from mvsnet_local.geometry import backproject, reproject, voxel_average


def ply(path):
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if line.startswith(b"element vertex "):
                count = int(line.split()[-1])
            if line == b"end_header\n":
                break
            assert line, path
        dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
        values = np.fromfile(f, dtype=dtype)
    assert len(values) == count, (path, len(values), count)
    xyz = np.column_stack([values[k] for k in "xyz"])
    color = np.column_stack([values[k] for k in "rgb"])
    assert np.isfinite(xyz).all(), path
    return xyz, color


def main():
    pairs = read_pairs()
    assert len(pairs) == 49
    summary = ROOT / "outputs/summary"
    results = json.loads((summary / "results.json").read_text())
    assert len(results["prediction"]) == 2 and len(results["fusion"]) == 6
    checkpoint_hash = hashlib.sha256(
        Path(str(CHECKPOINT) + ".data-00000-of-00001").read_bytes()
    ).hexdigest()
    confidence_edge_cases = []
    counts = {"prediction_maps": 0, "raw_pointclouds": 0, "fused_pointclouds": 0, "masks": 0}
    for views in [3, 5]:
        root = ROOT / "outputs" / f"views{views}"
        pred = root / "prediction"
        config = json.loads((pred / "config.json").read_text())
        assert (config["width"], config["height"], config["max_d"], config["interval_scale"]) == (
            1152,
            864,
            192,
            1.06,
        )
        assert config["checkpoint_sha256"] == checkpoint_hash and config["views"] == views
        data = {i: dict(np.load(pred / f"{i:08d}.npz")) for i in pairs}
        confidence = []
        fractions = []
        raw_count = 0
        all_raw_positions = []
        all_raw_colors = []
        retained = {t: 0 for t in [0.6, 0.8, 0.95]}
        fusion_rows = {
            t: list(csv.DictReader(open(root / f"fusion/prob_{round(t*100):03d}/metrics.csv")))
            for t in retained
        }
        positions = {t: [] for t in retained}
        colors = {t: [] for t in retained}
        for index, ref in data.items():
            assert ref["depth"].shape == (216, 288) and ref["rgb"].shape == (216, 288, 3)
            assert ref["source_ids"].tolist() == pairs[index][: views - 1]
            assert np.isfinite(ref["depth"]).all() and np.isfinite(ref["confidence"]).all()
            assert (
                ref["depth"].min() >= ref["depth_values"][0] - 0.001
                and ref["depth"].max() <= ref["depth_values"][-1] + 0.001
            )
            # 上游置信度是四项概率和；整数深度索引或端点会重复取值。
            assert ref["confidence"].min() >= 0 and ref["confidence"].max() <= 4.0001
            unusual = ref["confidence"] > 1.0001
            if unusual.any():
                confidence_edge_cases.append(
                    {
                        "views": views,
                        "view": index,
                        "pixels_above_one": int(unusual.sum()),
                        "maximum": float(ref["confidence"].max()),
                    }
                )
            np.testing.assert_array_equal(read_pfm(pred / f"{index:08d}_init.pfm"), ref["depth"])
            np.testing.assert_array_equal(
                read_pfm(pred / f"{index:08d}_prob.pfm"), ref["confidence"]
            )
            counts["prediction_maps"] += 1
            xyz, rgb = ply(root / f"raw/{index:08d}.ply")
            expected = backproject(ref["depth"], ref["K"], ref["E"])
            np.testing.assert_allclose(xyz, expected, atol=5e-5, rtol=1e-6)
            np.testing.assert_array_equal(rgb, ref["rgb"].reshape(-1, 3))
            raw_count += len(xyz)
            all_raw_positions.append(xyz)
            all_raw_colors.append(rgb)
            counts["raw_pointclouds"] += 1
            consistency = np.zeros(ref["depth"].shape, np.uint8)
            depth_sum = ref["depth"].copy()
            for source in pairs[index]:
                mask, returned, *_ = reproject(ref, data[source])
                consistency += mask
                depth_sum += np.where(mask, returned, 0)
            average = depth_sum / (consistency + 1)
            previous = np.ones(ref["depth"].shape, bool)
            for t in retained:
                folder = root / f"fusion/prob_{round(t*100):03d}"
                m = np.load(folder / f"masks/{index:08d}.npz")
                expected_mask = (ref["confidence"] > t) & (consistency >= 2)
                np.testing.assert_array_equal(m["photometric"], ref["confidence"] > t)
                np.testing.assert_array_equal(m["consistent_sources"], consistency)
                np.testing.assert_array_equal(m["final"], expected_mask)
                np.testing.assert_allclose(m["averaged_depth"], average, atol=1e-5)
                assert not (m["final"] & ~previous).any()
                previous = m["final"]
                kept = int(expected_mask.sum())
                retained[t] += kept
                counts["masks"] += 1
                row = next(r for r in fusion_rows[t] if int(r["view"]) == index)
                assert int(row["retained_points"]) == kept
                positions[t].append(backproject(average, ref["K"], ref["E"])[expected_mask.ravel()])
                colors[t].append(ref["rgb"].reshape(-1, 3)[expected_mask.ravel()])
            confidence.append(ref["confidence"].mean())
            fractions.append((ref["confidence"] > 0.8).mean())
        all_xyz, all_rgb = ply(root / "raw/all.ply")
        assert len(all_xyz) == raw_count
        np.testing.assert_array_equal(all_xyz, np.concatenate(all_raw_positions))
        np.testing.assert_array_equal(all_rgb, np.concatenate(all_raw_colors))
        counts["raw_pointclouds"] += 1
        means = next(r for r in results["prediction"] if r["views"] == views)
        np.testing.assert_allclose(means["mean_confidence"], np.mean(confidence), atol=1e-7)
        np.testing.assert_allclose(means["fraction_prob_08"], np.mean(fractions), atol=1e-10)
        for t in retained:
            xyz, color = ply(root / f"fusion/prob_{round(t*100):03d}/fused.ply")
            expected, expected_color = voxel_average(
                np.concatenate(positions[t]), np.concatenate(colors[t]), 0.5
            )
            np.testing.assert_allclose(xyz, expected, atol=1e-5)
            np.testing.assert_array_equal(color, expected_color)
            group = next(
                r for r in results["fusion"] if r["views"] == views and r["prob_threshold"] == t
            )
            assert (
                group["retained_observations"] == retained[t]
                and group["fused_points"] == len(xyz)
                and group["raw_points"] == raw_count
            )
            counts["fused_pointclouds"] += 1
    assert len(list(csv.DictReader(open(summary / "prediction_metrics.csv")))) == 98
    assert len(list(csv.DictReader(open(summary / "fusion_metrics.csv")))) == 294
    assert json.loads((summary / "model_parity.json").read_text())["passed"]
    assert json.loads((summary / "geometry_checks.json").read_text())["passed"]
    output = {
        "passed": True,
        "counts": counts,
        "upstream_confidence_edge_cases": confidence_edge_cases,
        "checkpoint_sha256": checkpoint_hash,
        "checks": [
            "all PFM/NPZ equality",
            "all raw PLY backprojections and colors",
            "all 980 source reprojections",
            "all masks and averaged depths",
            "all six fused PLY positions and colors after voxel merging",
            "confidence threshold nesting",
            "prediction/fusion table row counts and aggregate values",
        ],
    }
    save_json(summary / "verification.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
