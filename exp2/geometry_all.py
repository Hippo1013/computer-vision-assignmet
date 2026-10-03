#!/usr/bin/env python3
"""生成全部视角的直接点云和多视角后处理点云。"""
import argparse, csv, json, time
from pathlib import Path
import cv2
import numpy as np
from mvsnet_local.io import ROOT, read_pairs, write_ply, save_json
from mvsnet_local.geometry import backproject, reproject, voxel_average


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=int, nargs="+", default=[3, 5], choices=[3, 5])
    parser.add_argument("--prob-threshold", type=float, nargs="+", default=[0.6, 0.8, 0.95])
    parser.add_argument("--pixel-threshold", type=float, default=1.0)
    parser.add_argument("--depth-threshold", type=float, default=0.01)
    parser.add_argument(
        "--min-consistent",
        type=int,
        default=2,
        help="至少通过检查的源视图数量，2 表示包括参考图共 3 视图",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.5, help="毫米；最终点云的体素合并尺寸"
    )
    args = parser.parse_args()
    if any(t < 0 or t > 1 for t in args.prob_threshold):
        parser.error("概率阈值须在 [0,1] 内")
    pairs = read_pairs()
    for views in args.views:
        root = ROOT / "outputs" / f"views{views}"
        pred = root / "prediction"
        missing = [i for i in pairs if not (pred / f"{i:08d}.npz").exists()]
        if missing:
            raise RuntimeError(f"{pred} 缺少 {len(missing)} 个参考视角，请先完成 run.py")
        config = json.loads((pred / "config.json").read_text())
        data = {i: dict(np.load(pred / f"{i:08d}.npz")) for i in pairs}
        raw = root / "raw"
        raw.mkdir(exist_ok=True)
        all_xyz = []
        all_rgb = []
        rows = []
        groups = {}
        for threshold in args.prob_threshold:
            folder = root / "fusion" / f"prob_{round(threshold*100):03d}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "masks").mkdir(exist_ok=True)
            save_json(
                folder / "config.json",
                {
                    "prob_threshold": threshold,
                    "pixel_threshold": args.pixel_threshold,
                    "relative_depth_threshold": args.depth_threshold,
                    "min_consistent_sources": args.min_consistent,
                    "candidate_sources": 10,
                    "voxel_size_mm": args.voxel_size,
                    "input_views": views,
                    "source_confidence_filter": False,
                    "pixel_center": 0.5,
                    "prediction_config": config,
                    "method": "CPU multi-view reprojection + depth average + voxel average; not fusibile",
                },
            )
            groups[threshold] = {"folder": folder, "xyz": [], "rgb": [], "rows": []}
        for index, ref in data.items():
            t = time.perf_counter()
            xyz = backproject(ref["depth"], ref["K"], ref["E"])
            rgb = ref["rgb"].reshape(-1, 3)
            raw_valid = np.isfinite(xyz).all(1) & (ref["depth"].reshape(-1) > 0)
            write_ply(raw / f"{index:08d}.ply", xyz[raw_valid], rgb[raw_valid])
            all_xyz.append(xyz[raw_valid])
            all_rgb.append(rgb[raw_valid])
            count = np.zeros(ref["depth"].shape, np.uint8)
            depth_sum = ref["depth"].copy()
            pixel_sum = np.zeros_like(depth_sum)
            valid_count = np.zeros_like(count)
            for source_id in pairs[index]:
                if source_id not in data:
                    continue
                mask, returned, pixel_error, relative_error, valid = reproject(
                    ref, data[source_id], args.pixel_threshold, args.depth_threshold
                )
                count += mask
                depth_sum += np.where(mask, returned, 0)
                pixel_sum += np.where(valid, pixel_error, 0)
                valid_count += valid
            averaged = depth_sum / (count + 1)
            filtered_xyz = backproject(averaged, ref["K"], ref["E"])
            residual = np.divide(
                pixel_sum, valid_count, out=np.full_like(pixel_sum, np.nan), where=valid_count > 0
            )
            row = {
                "view": index,
                "raw_points": int(raw_valid.sum()),
                "geometric_points": int((count >= args.min_consistent).sum()),
                "median_all_pair_reprojection_px": float(np.nanmedian(residual)),
            }
            rows.append(row)
            for threshold, group in groups.items():
                photo = ref["confidence"] > threshold
                final = photo & (count >= args.min_consistent)
                group["xyz"].append(filtered_xyz[final.ravel()])
                group["rgb"].append(rgb[final.ravel()])
                np.savez_compressed(
                    group["folder"] / "masks" / f"{index:08d}.npz",
                    photometric=photo,
                    consistent_sources=count,
                    final=final,
                    averaged_depth=averaged,
                )
                cv2.imwrite(
                    str(group["folder"] / "masks" / f"{index:08d}.png"),
                    final.astype(np.uint8) * 255,
                )
                finite = residual[final & np.isfinite(residual)]
                group["rows"].append(
                    {
                        "view": index,
                        "raw_points": int(raw_valid.sum()),
                        "photometric_points": int(photo.sum()),
                        "geometric_points": row["geometric_points"],
                        "retained_points": int(final.sum()),
                        "retained_fraction": float(final.mean()),
                        "mean_confidence": (
                            float(ref["confidence"][final].mean()) if final.any() else None
                        ),
                        "median_all_pair_reprojection_px": (
                            float(np.median(finite)) if len(finite) else None
                        ),
                    }
                )
            print(
                f"views{views} {index:08d}: 几何保留 {(count>=args.min_consistent).mean():.1%}, {time.perf_counter()-t:.2f}s",
                flush=True,
            )
        write_ply(raw / "all.ply", np.concatenate(all_xyz), np.concatenate(all_rgb))
        with open(raw / "metrics.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        for threshold, group in groups.items():
            points = np.concatenate(group["xyz"])
            colors = np.concatenate(group["rgb"])
            if not len(points):
                raise RuntimeError(f"views{views} threshold {threshold}: 空点云")
            before = len(points)
            points, colors = voxel_average(points, colors, args.voxel_size)
            write_ply(group["folder"] / "fused.ply", points, colors)
            with open(group["folder"] / "metrics.csv", "w") as f:
                writer = csv.DictWriter(f, fieldnames=group["rows"][0])
                writer.writeheader()
                writer.writerows(group["rows"])
            save_json(
                group["folder"] / "summary.json",
                {
                    "views": views,
                    "prob_threshold": threshold,
                    "reference_views": len(data),
                    "raw_points": sum(r["raw_points"] for r in rows),
                    "retained_observations": before,
                    "fused_points": len(points),
                    "retained_fraction": before / sum(r["raw_points"] for r in rows),
                    "mean_view_reprojection_px": float(
                        np.mean(
                            [
                                r["median_all_pair_reprojection_px"]
                                for r in group["rows"]
                                if r["median_all_pair_reprojection_px"] is not None
                            ]
                        )
                    ),
                },
            )
    from summarize_results import main as summarize

    summarize()


if __name__ == "__main__":
    main()
