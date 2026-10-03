#!/usr/bin/env python3
"""深度预测：python run.py --views 3 / --views 5。"""
import argparse, hashlib, json, time, platform
from pathlib import Path
import cv2
import numpy as np
from mvsnet_local.io import ROOT, DATA, CHECKPOINT, read_pairs, load_view, write_pfm, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=int, default=5, choices=range(2, 12))
    p.add_argument("--width", type=int, default=1152)
    p.add_argument("--height", type=int, default=864)
    p.add_argument("--max-d", type=int, default=192)
    p.add_argument("--interval-scale", type=float, default=1.06)
    p.add_argument("--ids", type=int, nargs="+", help="仅运行指定参考视角；默认全部 49 个")
    p.add_argument("--outdir", type=Path)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.width % 32 or args.height % 32 or args.max_d % 8:
        p.error("宽高需为 32 的倍数，深度采样数需为 8 的倍数")
    if args.interval_scale <= 0:
        p.error("interval-scale 必须为正")
    if not args.outdir and (args.width, args.height, args.max_d, args.interval_scale) != (
        1152,
        864,
        192,
        1.06,
    ):
        p.error("修改默认尺寸或深度范围时请显式指定 --outdir，避免混入正式对比")
    out = args.outdir or ROOT / "outputs" / f"views{args.views}" / "prediction"
    out.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs(args.data / "pair.txt")
    ids = args.ids or list(pairs)
    if set(ids) - pairs.keys():
        p.error("ids 必须来自 pair.txt 的参考视角")
    checkpoint_hash = hashlib.sha256(
        Path(str(args.checkpoint) + ".data-00000-of-00001").read_bytes()
    ).hexdigest()
    config = {
        "views": args.views,
        "width": args.width,
        "height": args.height,
        "max_d": args.max_d,
        "interval_scale": args.interval_scale,
        "checkpoint_sha256": checkpoint_hash,
        "data": str(args.data.resolve()),
        "backend": "TensorFlow CPU",
        "normalization": "upstream GN / current-volume BN",
        "refinement": False,
        "depth_unit": "mm",
        "pixel_center": 0.5,
    }
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("输出目录已有其他设置；请使用新目录。")
    save_json(config_path, config)
    from mvsnet_local.model import MVSNet, tf

    model = MVSNet(args.checkpoint, args.threads)
    timings = {}
    timing_path = out / "inference_timing.json"
    if timing_path.exists():
        timings = json.loads(timing_path.read_text())["per_view"]
    cache = {}

    def get(index):
        if index not in cache:
            t = time.perf_counter()
            image, rgb, K, E, start, interval = load_view(index, args.width, args.height, args.data)
            feat = model.features(image[None])
            feat.numpy()
            cache[index] = (feat, rgb, K, E, start, interval, time.perf_counter() - t)
        return cache[index]

    for position, index in enumerate(ids):
        file = out / f"{index:08d}.npz"
        if file.exists() and str(index) in timings and not args.overwrite:
            print(f"[{position+1}/{len(ids)}] {index:08d} 已有结果，跳过", flush=True)
            continue
        selected = [index] + pairs[index][: args.views - 1]
        t = time.perf_counter()
        data = [get(j) for j in selected]
        start, interval = data[0][4:6]
        depths = start + np.arange(args.max_d, dtype=np.float32) * interval * args.interval_scale
        inference_start = time.perf_counter()
        depth, confidence = model.predict(
            [d[0] for d in data], [(d[2], d[3]) for d in data], depths
        )
        matching = time.perf_counter() - inference_start
        elapsed = time.perf_counter() - t
        assert np.isfinite(depth).all() and np.isfinite(confidence).all()
        np.savez_compressed(
            file,
            depth=depth,
            confidence=confidence,
            rgb=data[0][1],
            K=data[0][2],
            E=data[0][3],
            source_ids=np.array(selected[1:]),
            depth_values=depths,
        )
        write_pfm(out / f"{index:08d}_init.pfm", depth)
        write_pfm(out / f"{index:08d}_prob.pfm", confidence)
        cv2.imwrite(
            str(out / f"{index:08d}_depth.png"),
            cv2.applyColorMap(
                np.uint8(np.clip((depth - 425) / (935 - 425), 0, 1) * 255), cv2.COLORMAP_TURBO
            ),
        )
        cv2.imwrite(str(out / f"{index:08d}_prob.png"), np.uint8(np.clip(confidence, 0, 1) * 255))
        cv2.imwrite(str(out / f"{index:08d}_rgb.png"), cv2.cvtColor(data[0][1], cv2.COLOR_RGB2BGR))
        timings[str(index)] = {
            "source_ids": selected[1:],
            "matching_seconds": matching,
            "wall_seconds": elapsed,
            "feature_seconds_sum": sum(d[6] for d in data),
            "mean_confidence": float(confidence.mean()),
            "fraction_prob_08": float((confidence > 0.8).mean()),
            "depth_min": float(depth.min()),
            "depth_max": float(depth.max()),
        }
        save_json(
            timing_path,
            {
                "python": platform.python_version(),
                "tensorflow": tf.__version__,
                "per_view": timings,
                "note": "matching_seconds 含单应变换、代价体、3D CNN 和输出；特征按视角缓存，首次调用含图编译。",
            },
        )
        print(
            f"[{position+1}/{len(ids)}] {index:08d}: {matching:.2f}s, p>.8={(confidence>.8).mean():.1%}",
            flush=True,
        )
    if model.used:
        save_json(
            out / "weights_used.json",
            {"checkpoint_sha256": checkpoint_hash, "tensor_names": sorted(model.used)},
        )
    print(f"输出：{out}", flush=True)


if __name__ == "__main__":
    main()
