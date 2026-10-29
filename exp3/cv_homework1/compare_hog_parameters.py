"""Requirement 2: preserve HOG arrays and generate material for joint review.

Uses skimage's installed private histogram routine only to recover the intermediate
Cell histograms. Reconstruction of BOTH public outputs is checked for every run.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import html
import inspect
import json
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import draw
from skimage.feature import hog
from skimage.feature import _hog as implementation


ROOT = Path(__file__).resolve().parent
GROUPS = {
    "orientations": [(4, 8, 2), (9, 8, 2), (18, 8, 2)],
    "cell_size": [(9, 4, 2), (9, 8, 2), (9, 16, 2)],
    "block_size": [(9, 8, 1), (9, 8, 2), (9, 8, 4)],
}
GROUP_TITLES = {
    "orientations": "方向数对比",
    "cell_size": "Cell大小对比",
    "block_size": "Block大小对比",
}
# Pixel coordinates are (row, col), zero based. ROI is [y0:y1, x0:x1].
ANCHOR = (80, 432)
ROI = (32, 160, 352, 480)


def name(config):
    o, p, b = config
    return f"o{o}_p{p}_b{b}"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def recover_histogram(gray, orientations, p):
    """Match the exact installed HOG gradient and Cython histogram implementation."""
    gy, gx = implementation._hog_channel_gradient(gray.astype(np.float64))
    h, w = gray.shape
    hist = np.zeros((h // p, w // p, orientations), dtype=np.float64)
    implementation._hoghistogram.hog_histograms(
        gx, gy, p, p, w, h, w // p, h // p, orientations, hist
    )
    return hist


def reconstruct_visualization(hist, image_shape, p):
    vis = np.zeros(image_shape, dtype=np.float64)
    radius = p // 2 - 1
    angles = np.pi * (np.arange(hist.shape[-1]) + 0.5) / hist.shape[-1]
    for r in range(hist.shape[0]):
        for c in range(hist.shape[1]):
            cy, cx = r * p + p // 2, c * p + p // 2
            for o, angle in enumerate(angles):
                dy, dx = radius * np.sin(angle), radius * np.cos(angle)
                rr, cc = draw.line(int(cy - dx), int(cx + dy),
                                   int(cy + dx), int(cx - dy))
                vis[rr, cc] += hist[r, c, o]
    return vis


def run_config(gray, config, output):
    o, p, b = config
    folder = output / name(config)
    folder.mkdir()
    params = dict(orientations=o, pixels_per_cell=(p, p),
                  cells_per_block=(b, b), block_norm="L2-Hys",
                  transform_sqrt=False, channel_axis=None)
    blocks, vis = hog(gray, **params, visualize=True, feature_vector=False)
    vector, vis_flat_call = hog(gray, **params, visualize=True, feature_vector=True)
    hist = recover_histogram(gray, o, p)
    reconstructed = np.empty_like(blocks)
    for r in range(blocks.shape[0]):
        for c in range(blocks.shape[1]):
            reconstructed[r, c] = implementation._hog_normalize_block(
                hist[r:r + b, c:c + b], method="L2-Hys"
            )
    expected = (gray.shape[0] // p - b + 1) * (gray.shape[1] // p - b + 1) * b*b*o
    checks = {
        "dimension_matches_formula": vector.size == expected,
        "flat_and_structured_outputs_exact": np.array_equal(vector, blocks.ravel()),
        "two_public_visualizations_exact": np.array_equal(vis, vis_flat_call),
        "histogram_reconstructs_all_blocks_exact": np.array_equal(reconstructed, blocks),
        "histogram_reconstructs_visualization_exact": np.array_equal(
            reconstruct_visualization(hist, gray.shape, p), vis),
        "all_arrays_finite": all(np.isfinite(x).all() for x in (vector, vis, hist)),
    }
    if not all(checks.values()):
        raise AssertionError(f"{name(config)}: {checks}")
    cy, cx = ANCHOR[0] // p, ANCHOR[1] // p
    # Every occurrence of this physical Cell, with its enclosing Block index.
    contexts, indices = [], []
    for r in range(max(0, cy-b+1), min(cy+1, blocks.shape[0])):
        for c in range(max(0, cx-b+1), min(cx+1, blocks.shape[1])):
            contexts.append(blocks[r, c, cy-r, cx-c])
            indices.append((r, c))
    target = blocks[cy, cx, 0, 0].copy()
    np.save(folder / "features.npy", vector, allow_pickle=False)
    np.save(folder / "hog_vis.npy", vis, allow_pickle=False)
    np.save(folder / "cell_histograms.npy", hist, allow_pickle=False)
    np.savez_compressed(folder / "target_cell.npz", raw=hist[cy, cx],
                        normalized_in_anchored_block=target,
                        all_normalized_occurrences=np.asarray(contexts),
                        enclosing_block_indices=np.asarray(indices),
                        orientation_edges_deg=np.linspace(0, 180, o+1))
    edges = np.linspace(0, 1, 101)
    counts, _ = np.histogram(vector, bins=edges)
    if int(counts.sum()) != vector.size:
        raise AssertionError("Feature values outside the shared [0, 1] bins")
    np.savez_compressed(folder / "feature_distribution.npz", edges=edges, counts=counts,
                        proportions=counts/vector.size)
    with (folder / "orientation_histograms.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["angle_start_deg", "angle_end_deg", "target_cell_raw",
                         "all_cells_mean_raw", "target_cell_normalized_in_anchored_block"])
        for i in range(o):
            writer.writerow([180*i/o, 180*(i+1)/o, hist[cy, cx, i],
                             hist[..., i].mean(), target[i]])
    record = {
        "id": name(config), "config": list(config), "parameters": params,
        "cell_histogram_shape": list(hist.shape), "block_shape": list(blocks.shape),
        "feature_dimension": int(vector.size), "feature_dtype": str(vector.dtype),
        "feature_bytes": int(vector.nbytes), "block_pixel_size": [p*b, p*b],
        "target_cell_index": [cy, cx], "target_cell_pixel_box_yxyx": [cy*p, cx*p, (cy+1)*p, (cx+1)*p],
        "anchored_block_index": [cy, cx],
        "target_cell_occurrences": len(contexts),
        "feature_statistics": {
            "min": float(vector.min()), "max": float(vector.max()),
            "mean": float(vector.mean()), "std": float(vector.std()),
            "zero_fraction": float(np.mean(vector == 0)),
            "l2_norm": float(np.linalg.norm(vector)),
        }, "checks": checks,
    }
    write_json(folder / "metadata.json", record)
    return {"record": record, "vis": vis, "hist": hist, "target": target,
            "counts": counts, "vector": vector, "blocks": blocks}


def plot_group(group, configs, data, gray, output, vmax):
    fig, axes = plt.subplots(4, 3, figsize=(13, 13), layout="constrained",
                             gridspec_kw={"height_ratios": [0.8, 1.15, 0.85, 0.85]})
    raw_max = max(data[name(c)]["hist"][ANCHOR[0]//c[1], ANCHOR[1]//c[1]].max() for c in configs)
    normalized_max = max(data[name(c)]["target"].max() for c in configs)
    y0, y1, x0, x1 = ROI
    for col, config in enumerate(configs):
        o, p, b = config
        result = data[name(config)]
        axes[0, col].imshow(np.sqrt(result["vis"]), cmap="gray", vmin=0, vmax=vmax)
        axes[0, col].add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec="#ffb454", lw=1))
        axes[0, col].set_title(f"o={o}, Cell={p}×{p}, Block={b}×{b}\nD={result['record']['feature_dimension']:,}")
        axes[0, col].axis("off")
        axes[1, col].imshow(np.sqrt(result["vis"][y0:y1, x0:x1]), cmap="gray", vmin=0, vmax=vmax,
                             extent=(x0, x1, y1, y0), interpolation="nearest")
        cy, cx = result["record"]["target_cell_index"]
        axes[1, col].add_patch(plt.Rectangle((cx*p, cy*p), p, p, fill=False, ec="#ffb454", lw=1.5))
        axes[1, col].add_patch(plt.Rectangle((cx*p, cy*p), b*p, b*p, fill=False,
                                           ec="#4dd7d0", ls="--", lw=1))
        axes[1, col].set_title("同一局部区域；橙框：目标Cell；青框：归一化Block", fontsize=9)
        axes[1, col].set_xlabel("x / pixel")
        centers = (np.arange(o)+0.5)*180/o
        raw = result["hist"][cy, cx]
        axes[2, col].bar(centers, raw, width=180/o*0.85, color="#087e88")
        axes[2, col].set(title="目标Cell方向直方图（Block归一化前）", xlim=(0, 180), ylim=(0, raw_max*1.12),
                         xlabel="梯度方向 / °", ylabel="原始统计值")
        axes[3, col].bar(centers, result["target"], width=180/o*0.85, color="#5368b5")
        axes[3, col].set(title="目标Cell特征（在青框Block中归一化后）", xlim=(0, 180),
                         ylim=(0, normalized_max*1.12), xlabel="梯度方向 / °", ylabel="特征值")
        for ax in axes[2:, col]:
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{GROUP_TITLES[group]}｜统一显示范围；显示用平方根变换，原始数组完整保存", fontsize=15)
    fig.savefig(output / f"{group}_comparison.png", dpi=160)
    plt.close(fig)


def make_review(output, data, comparisons, provenance, vmax):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial Unicode MS", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    gray = cv2.imread(str(output / "input_gray.png"), cv2.IMREAD_GRAYSCALE)
    for group, configs in GROUPS.items():
        plot_group(group, configs, data, gray, output, vmax)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), layout="constrained")
    for ax, (group, configs) in zip(axes, GROUPS.items()):
        for config in configs:
            result = data[name(config)]
            ax.stairs(result["counts"]/result["vector"].size, np.linspace(0, 1, 101), label=name(config))
        ax.set(title=GROUP_TITLES[group], xlabel="最终特征值（共同的0.01宽度分箱）", ylabel="特征元素比例", xlim=(0, 1))
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("完整特征的数值分布｜维度和元素语义不同，不做展开向量的逐项差分")
    fig.savefig(output / "feature_distributions.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    ax.imshow(cv2.cvtColor(cv2.imread(str(output / "input_color.png")), cv2.COLOR_BGR2RGB))
    y0, y1, x0, x1 = ROI
    ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec="#ffb454", lw=2))
    ax.scatter([ANCHOR[1]], [ANCHOR[0]], marker="+", s=100, c="#4dd7d0")
    ax.set_title("输入图像与固定审阅区域｜橙框为128×128局部区域；十字为目标Cell定位点")
    ax.axis("off")
    fig.savefig(output / "input_review.png", dpi=160)
    plt.close(fig)
    rows = []
    for result in data.values():
        record = result["record"]
        o, p, b = record["config"]
        rows.append(f"<tr><td>{record['id']}</td><td>{o}</td><td>{p}×{p}</td><td>{b}×{b}</td>"
                    f"<td>{record['feature_dimension']:,}</td><td>{record['feature_bytes']/1048576:.2f} MiB</td>"
                    f"<td><a href='{record['id']}/features.npy'>完整特征</a> · "
                    f"<a href='{record['id']}/hog_vis.npy'>可视化数组</a> · "
                    f"<a href='{record['id']}/cell_histograms.npy'>Cell直方图</a> · "
                    f"<a href='{record['id']}/orientation_histograms.csv'>直方图CSV</a> · "
                    f"<a href='{record['id']}/metadata.json'>参数与核验</a></td></tr>")
    sections = "".join(f"<section><h2>{GROUP_TITLES[g]}</h2><img src='{g}_comparison.png' loading='lazy'></section>" for g in GROUPS)
    block_checks = "".join(f"<li>{x['pair'][0]} 与 {x['pair'][1]}：可视化逐像素相同={x['visualizations_exact']}；"
                          f"Cell原始直方图相同={x['raw_cell_histograms_exact']}；"
                          f"同一目标Cell在指定Block中的归一化向量L2差={x['anchored_target_l2_difference']:.6f}</li>"
                          for x in comparisons)
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HOG参数实验结果审阅</title><style>body{{font:16px/1.7 system-ui,sans-serif;color:#20313e;background:#f4f7f9;margin:0 auto;padding:30px;max-width:1400px}}section{{background:white;padding:24px;margin:24px 0;border-radius:12px}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}a{{color:#087e88}}code{{background:#edf1f5;padding:2px 5px}}.note{{border-left:4px solid #087e88;padding:12px 20px;background:#eaf6f6}}</style>
<h1>HOG参数实验结果审阅</h1><p>同一幅432×768灰度图，7组配置；仅供实验结果讨论，尚未写入课程报告。</p>
<div class="note">完整最终特征、HOG可视化原始数组与Block归一化前的Cell方向直方图均已保存。灰度化与原visualize_hog.py一致，未缩放、未增强。</div>
<section><h2>输入与观察区域</h2><img src="input_review.png"><p>固定ROI为行32:160、列352:480；目标像素为(y=80,x=432)。不同Cell大小覆盖的物理范围不同，详见各配置metadata.json；Block对比中目标Cell相同，所画归一化Block均以该Cell为左上角。</p></section>
<section><h2>配置与完整数据</h2><table><tr><th>配置</th><th>方向数</th><th>Cell/像素</th><th>Block/Cell</th><th>特征维度</th><th>特征内存</th><th>数据文件</th></tr>{''.join(rows)}</table>
<p>features.npy为公开hog函数返回的一维float64完整特征；按metadata.json的block_shape重排即可恢复五维Block数组。cell_histograms.npy形状为(Cell行数,Cell列数,方向数)，是当前实现的梯度幅值加权统计，不是简单像素个数。</p></section>
{sections}<section><h2>完整特征数值分布</h2><img src="feature_distributions.png"><p>使用共同的[0,1]分箱和元素比例，纵轴为对数坐标（零频率不显示）。这些分布只反映数值构成，不证明哪组分类更好；不同配置的展开向量不具有统一的逐元素语义。</p></section>
<section><h2>Block组数值检查</h2><ul>{block_checks}</ul><p>这里只比较同一个Cell在明确指定的Block中的同维方向向量。每个配置target_cell.npz另保存该Cell在所有覆盖Block中的归一化结果与Block索引。</p></section>
<section><h2>显示与核验说明</h2><p>全部HOG图采用相同的平方根显示变换及同一亮度范围；显示上限为全部7组平方根图像值合并后的99.5百分位（{vmax:.6f}），高于上限的像素仅在PNG显示中截断，原始.npy不变。图中的线段是HOG的绘图表示，不能直接当作最终归一化特征。</p>
<p>恢复中间直方图使用当前scikit-image私有实现；逐组验证它能逐元素精确重建公开函数的全部Block特征和原始可视化图。已检查两种feature_vector设置、维度公式、有限值和落盘数组一致性。<a href="verification.json">核验记录</a> · <a href="results.json">完整指标</a> · <a href="provenance.json">实验协议</a></p></section></html>"""
    (output / "review.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "jinx.jpg")
    parser.add_argument("--output", type=Path, default=None,
                        help="New/empty output directory; existing runs are never overwritten")
    args = parser.parse_args()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    output = (args.output or ROOT / "outputs" / f"hog_parameter_review_{now:%Y%m%d_%H%M%S}").resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    color = cv2.imread(str(args.image.resolve()))
    if color is None:
        raise FileNotFoundError(args.image)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    if gray.shape != (432, 768):
        raise ValueError(f"Fixed ROI protocol requires 432×768 input, got {gray.shape}")
    cv2.imwrite(str(output / "input_color.png"), color)
    cv2.imwrite(str(output / "input_gray.png"), gray)
    gy, gx = implementation._hog_channel_gradient(gray.astype(np.float64))
    np.savez_compressed(output / "gradients.npz", row=gy, col=gx,
                        magnitude=np.hypot(gy, gx), unsigned_angle_deg=np.rad2deg(np.arctan2(gy, gx)) % 180)
    provenance = {
        "created_at": now.isoformat(), "input": str(args.image.resolve()),
        "input_sha256": sha(args.image), "image_shape": list(gray.shape),
        "python": platform.python_version(), "numpy": np.__version__,
        "skimage": skimage.__version__, "opencv": cv2.__version__, "matplotlib": matplotlib.__version__,
        "script_sha256": sha(__file__), "hog_python_sha256": sha(implementation.__file__),
        "hog_histogram_binary_sha256": sha(implementation._hoghistogram.__file__),
        "preprocessing": "cv2.imread(BGR) -> cv2.cvtColor(COLOR_BGR2GRAY); uint8; no /255, resize or augmentation",
        "fixed_parameters": {"block_norm": "L2-Hys", "transform_sqrt": False, "channel_axis": None},
        "groups": GROUPS, "roi_y0_y1_x0_x1": ROI, "target_pixel_yx": ANCHOR,
        "normalized_target_context": "Block top-left is the target Cell; select its [0,0,:] vector",
        "intermediate_extraction": "Exact installed private gradient/Cython histogram routines; both public outputs reconstructed and checked",
        "histogram_semantics": "Block-unnormalized per-Cell histogram as used by installed hog; gradient magnitude weighted, not pixel counts",
    }
    data = {}
    for config in dict.fromkeys(c for group in GROUPS.values() for c in group):
        print(f"Extracting {name(config)}", flush=True)
        data[name(config)] = run_config(gray, config, output)
    sqrt_values = np.concatenate([np.sqrt(x["vis"]).ravel() for x in data.values()])
    vmax = float(np.percentile(sqrt_values, 99.5))
    provenance["display"] = {"transform": "sqrt", "vmin": 0, "vmax": vmax,
                              "vmax_rule": "pooled 99.5th percentile across all 7 images", "shared_across_all_configs": True}
    checks = []
    for config in GROUPS["block_size"]:
        if config == (9, 8, 2):
            continue
        baseline, other = data["o9_p8_b2"], data[name(config)]
        check = {"pair": ["o9_p8_b2", name(config)],
                 "visualizations_exact": bool(np.array_equal(baseline["vis"], other["vis"])),
                 "raw_cell_histograms_exact": bool(np.array_equal(baseline["hist"], other["hist"])),
                 "anchored_target_l2_difference": float(np.linalg.norm(baseline["target"]-other["target"]))}
        if not check["visualizations_exact"] or not check["raw_cell_histograms_exact"] or check["anchored_target_l2_difference"] == 0:
            raise AssertionError(check)
        checks.append(check)
    reloaded_ok = True
    for key, result in data.items():
        for filename, array in [("features.npy", result["vector"]), ("hog_vis.npy", result["vis"]),
                                ("cell_histograms.npy", result["hist"])]:
            reloaded_ok &= np.array_equal(np.load(output/key/filename, allow_pickle=False), array)
        plt.imsave(output/key/"hog_vis.png", np.sqrt(result["vis"]), cmap="gray", vmin=0, vmax=vmax)
    if not reloaded_ok:
        raise AssertionError("Saved arrays do not match memory")
    write_json(output / "provenance.json", provenance)
    records = [x["record"] for x in data.values()]
    write_json(output / "results.json", {"configurations": records, "block_comparisons": checks})
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["config", "orientations", "cell_side_px", "block_side_cells", "feature_dimension", "feature_bytes", "min", "max", "mean", "std", "zero_fraction"])
        for record in records:
            stats = record["feature_statistics"]
            writer.writerow([record["id"], *record["config"], record["feature_dimension"], record["feature_bytes"],
                             *(stats[k] for k in ["min", "max", "mean", "std", "zero_fraction"])])
    write_json(output / "verification.json", {"passed": True, "configuration_count": len(data),
               "all_configuration_checks": all(all(r["checks"].values()) for r in records),
               "saved_arrays_exact": bool(reloaded_ok), "block_comparisons": checks})
    make_review(output, data, checks, provenance, vmax)
    lines = ["# HOG参数实验审阅记录", "", "状态：实验已运行，尚未写入课程报告。", "",
             "审阅入口：review.html；三张参数对比图与feature_distributions.png。", "",
             "|配置|特征维度|float64特征大小|", "|---|---:|---:|"]
    lines += [f"|{r['id']}|{r['feature_dimension']:,}|{r['feature_bytes']/1048576:.2f} MiB|" for r in records]
    lines += ["", "原始完整数组：各配置features.npy、hog_vis.npy、cell_histograms.npy。",
              "features.npy按metadata.json内block_shape重排即可恢复五维Block数组；无需保存重复副本。",
              "target_cell.npz保存指定位置Cell的归一化前直方图及其在所有覆盖Block中的归一化向量。",
              "梯度幅值、方向和行列梯度保存在根目录gradients.npz。",
              "", "已验证：每组中间直方图能精确重建公开hog函数返回的全部Block特征和可视化；完整向量与结构化结果一致；落盘数组一致。",
              "Block组的原始Cell直方图和可视化完全相同，同一Cell在指定Block中的归一化向量不同。",
              "", "显示：全部图统一平方根变换与亮度范围，99.5百分位截断只用于PNG；.npy保持原值。",
              "不同参数的展开向量维度或元素语义不同，不做逐项差分；特征数值分布和单张图的观察不能说明分类优劣。",
              "目标Cell随Cell尺寸变化覆盖范围不同；Block组目标Cell位置和范围相同。",
              "", f"复现命令：python compare_hog_parameters.py --output <新的空目录>", ""]
    (output / "review.md").write_text("\n".join(lines))
    manifest = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "files.json", manifest)
    print(json.dumps({"output": str(output), "configs": len(data), "passed": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
