#!/usr/bin/env python3
"""从实验原始数组与 PLY 生成报告图表，不重新运行模型。"""
from pathlib import Path
import csv, json, sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mvsnet_local.geometry import backproject
from mvsnet_local.experiment import VIEW_COUNTS, REPORT_VIEWS, PROB_THRESHOLDS

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "axes.titlecolor": "#203C5D",
        "text.color": "#202B35",
    }
)


def read_ply(path):
    with open(path, "rb") as f:
        n = 0
        while True:
            line = f.readline()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            if line == b"end_header\n":
                break
            if not line:
                raise ValueError(path)
        a = np.fromfile(
            f,
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
            count=n,
        )
    return np.column_stack([a[k] for k in ["x", "y", "z"]]), np.column_stack(
        [a[k] for k in ["r", "g", "b"]]
    )


def cloud_project(xyz, ref):
    camera = (ref["E"][:3, :3] @ xyz.T + ref["E"][:3, 3:4]).T
    # 从参考图方向略向侧面转动，显示点云的空间厚度。
    angle = np.deg2rad(-22)
    rotation = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )
    center = np.array([0.0, 0.0, 650.0])
    return (camera - center) @ rotation.T


def cloud_image(xyz, rgb, ref, bounds, size=(1200, 900), radius=1):
    p = cloud_project(xyz, ref)
    w, h = size
    xmin, xmax, ymin, ymax = bounds
    scale = min((w - 50) / (xmax - xmin), (h - 50) / (ymax - ymin))
    x = np.rint((p[:, 0] - (xmin + xmax) / 2) * scale + w / 2).astype(int)
    y = np.rint((p[:, 1] - (ymin + ymax) / 2) * scale + h / 2).astype(int)
    good = (x >= radius) & (x < w - radius) & (y >= radius) & (y < h - radius)
    x = x[good]
    y = y[good]
    z = p[good, 2]
    rgb = rgb[good]
    # 每个像素保存最近点；点的大小和投影范围在所有组间相同。
    canvas = np.full((h * w, 3), 248, np.uint8)
    depth = np.full(h * w, np.inf)
    order = np.argsort(z)
    x = x[order]
    y = y[order]
    z = z[order]
    rgb = rgb[order]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            idx = (y + dy) * w + x + dx
            unique, first = np.unique(idx, return_index=True)
            front = z[first] < depth[unique]
            canvas[unique[front]] = rgb[first[front]]
            depth[unique[front]] = z[first[front]]
    return canvas.reshape(h, w, 3)


def table(filename, caption, headers, rows):
    text = (
        "\\ReportTableCaption{"
        + caption
        + "}\n\\begin{ReportTable}{@{}"
        + "C" * len(headers)
        + "@{}}\n\\rowcolor{ReportNavy}\n"
    )
    text += " & ".join("\\ReportHead{" + h + "}" for h in headers) + " \\\\\n"
    for row in rows:
        text += " & ".join(map(str, row)) + " \\\\\n"
    text += "\\end{ReportTable}\n"
    (FIG.parent / filename).write_text(text)


def main():
    results = json.loads((ROOT / "outputs/summary/results.json").read_text())
    expected_groups = {(v, t) for v in VIEW_COUNTS for t in PROB_THRESHOLDS}
    if {g["views"] for g in results["prediction"]} != set(VIEW_COUNTS) or {
        (g["views"], g["prob_threshold"]) for g in results["fusion"]
    } != expected_groups:
        raise RuntimeError("请先完成全部视图组预测和四档阈值点云")
    ids = [12, 25, 43]
    datasets = {
        v: {i: dict(np.load(ROOT / f"outputs/views{v}/prediction/{i:08d}.npz")) for i in ids}
        for v in VIEW_COUNTS
    }
    fig, axes = plt.subplots(3, 5, figsize=(15, 8.5), layout="constrained")
    for row, i in enumerate(ids):
        b = datasets[REPORT_VIEWS][i]
        for col, values in enumerate([b["rgb"], *[datasets[v][i]["depth"] for v in VIEW_COUNTS], b["confidence"]]):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(values)
            elif col in [1, 2, 3]:
                im = ax.imshow(values, cmap="turbo", vmin=425, vmax=935)
            else:
                pr = ax.imshow(values, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(["参考图像", *[f"{v} 视图深度" for v in VIEW_COUNTS], f"{REPORT_VIEWS} 视图置信度"][col])
            if col == 0:
                ax.set_ylabel(f"{i:08d}", fontsize=12)
    fig.colorbar(
        im,
        ax=axes[:, 1:4],
        orientation="horizontal",
        shrink=0.7,
        pad=0.015,
        label="相机 Z 深度 / mm",
    )
    fig.colorbar(pr, ax=axes[:, 4], orientation="horizontal", shrink=0.8, pad=0.015, label="置信度")
    fig.savefig(FIG / "depth_comparison.png", dpi=190)
    plt.close(fig)
    ref = datasets[REPORT_VIEWS][12]
    clouds = {
        t: read_ply(ROOT / f"outputs/views{REPORT_VIEWS}/fusion/prob_{round(t*100):03d}/fused.ply")
        for t in PROB_THRESHOLDS
    }
    xyz, rgb = clouds[PROB_THRESHOLDS[0]]
    p = cloud_project(xyz, ref)
    # 所有阈值图共享最低阈值组的全点云包围范围，不单独放大某组。
    lo = p[:, :2].min(0)
    hi = p[:, :2].max(0)
    padding = (hi - lo) * 0.035
    bounds = (lo[0] - padding[0], hi[0] + padding[0], lo[1] - padding[1], hi[1] + padding[1])
    fig, axes = plt.subplots(1, len(PROB_THRESHOLDS), figsize=(16, 4.3), layout="constrained")
    for ax, (t, (xyz, rgb)) in zip(axes, clouds.items()):
        im = cloud_image(xyz, rgb, ref, bounds)
        Image.fromarray(im).save(FIG / f"cloud_prob_{round(t*100):03d}.png")
        ax.imshow(im)
        ax.set_title(f"置信度阈值 {t:g}")
        ax.axis("off")
    fig.savefig(FIG / "threshold_clouds.png", dpi=200)
    plt.close(fig)
    # 固定参考视图的过滤掩膜，补充全局点云中不易看到的局部丢点。
    fig, axes = plt.subplots(1, len(PROB_THRESHOLDS), figsize=(16, 3.5), layout="constrained")
    for ax, t in zip(axes, PROB_THRESHOLDS):
        mask = np.load(ROOT / f"outputs/views{REPORT_VIEWS}/fusion/prob_{round(t*100):03d}/masks/00000012.npz")[
            "final"
        ]
        im = ref["rgb"].copy()
        im[~mask] = [237, 241, 245]
        ax.imshow(im)
        ax.set_title(f"00000012 · 阈值 {t:g}")
        ax.axis("off")
    fig.savefig(FIG / "threshold_masks.png", dpi=200)
    plt.close(fig)
    raw = read_ply(ROOT / f"outputs/views{REPORT_VIEWS}/raw/all.ply")
    filtered = clouds[0.8]
    p = cloud_project(raw[0], ref)
    lo = p[:, :2].min(0)
    hi = p[:, :2].max(0)
    padding = (hi - lo) * 0.02
    raw_bounds = (lo[0] - padding[0], hi[0] + padding[0], lo[1] - padding[1], hi[1] + padding[1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, (xyz, rgb), title in zip(axes, [raw, filtered], ["直接反投影", "后处理 · 阈值 0.8"]):
        ax.imshow(cloud_image(xyz, rgb, ref, raw_bounds))
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(FIG / "raw_vs_fused.png", dpi=210)
    plt.close(fig)
    # 固定使用与深度图相同的三个参考视角；每对图共享原始点云的完整范围。
    single_comparisons = []
    for image_id in ids:
        single_ref = datasets[REPORT_VIEWS][image_id]
        raw = read_ply(ROOT / f"outputs/views{REPORT_VIEWS}/raw/{image_id:08d}.ply")
        with np.load(ROOT / f"outputs/views{REPORT_VIEWS}/fusion/prob_080/masks/{image_id:08d}.npz") as masks:
            sel = masks["final"].ravel()
            filtered = (
                backproject(masks["averaged_depth"], single_ref["K"], single_ref["E"])[sel],
                single_ref["rgb"].reshape(-1, 3)[sel],
            )
        p = cloud_project(raw[0], single_ref)
        lo = p[:, :2].min(0)
        hi = p[:, :2].max(0)
        padding = (hi - lo) * 0.03
        single_bounds = (lo[0] - padding[0], hi[0] + padding[0], lo[1] - padding[1], hi[1] + padding[1])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        panels = []
        for ax, (xyz, rgb), title in zip(
            axes, [raw, filtered], [f"{image_id:08d} · 直接反投影", f"{image_id:08d} · 过滤与深度平均"]
        ):
            panel = cloud_image(xyz, rgb, single_ref, single_bounds)
            panels.append(panel)
            ax.imshow(panel)
            ax.set_title(title)
            ax.axis("off")
        filename = "single_raw_vs_filtered.png" if image_id == 12 else f"single_raw_vs_filtered_{image_id:04d}.png"
        fig.savefig(FIG / filename, dpi=200)
        plt.close(fig)
        single_comparisons.append((image_id, panels))
    fig, axes = plt.subplots(2, len(ids), figsize=(12, 6.8), layout="constrained")
    for col, (image_id, panels) in enumerate(single_comparisons):
        for row, (panel, label) in enumerate(zip(panels, ["直接反投影", "过滤与深度平均"])):
            axes[row, col].imshow(panel)
            axes[row, col].set_title(f"{image_id:04d} · {label}", fontsize=14)
            axes[row, col].axis("off")
    fig.savefig(FIG / "single_views_comparison.png", dpi=220)
    plt.close(fig)
    # 表格中的每一项都从正式输出计算。
    rows = []
    for group in results["prediction"]:
        v = group["views"]
        f = next(g for g in results["fusion"] if g["views"] == v and g["prob_threshold"] == 0.8)
        rows.append(
            [
                v,
                f"{group['mean_confidence']:.3f}",
                f"{group['fraction_prob_08']*100:.2f}\\%",
                f"{f['retained_fraction']*100:.2f}\\%",
                f"{f['fused_points']:,}",
            ]
        )
    table(
        "view-comparison-table.tex",
        "3、7、11 视图的统计结果（49 个参考视角）",
        ["输入视图数", "平均置信度", "$p>0.8$ 比例", "最终保留比例", "融合点数"],
        rows,
    )
    rows = []
    for f in results["fusion"]:
        if f["views"] == REPORT_VIEWS:
            rows.append(
                [
                    f"{f['prob_threshold']:g}",
                    f"{f['retained_observations']:,}",
                    f"{f['retained_fraction']*100:.2f}\\%",
                    f"{f['fused_points']:,}",
                    f"{f['mean_view_reprojection_px']:.3f}",
                ]
            )
    table(
        "threshold-table.tex",
        f"{REPORT_VIEWS} 视图组不同置信度阈值的点云统计",
        ["阈值", "保留观测点数", "保留比例", "融合点数", "重投影偏差 / px"],
        rows,
    )
    print("报告图表已更新：", FIG)


if __name__ == "__main__":
    main()
