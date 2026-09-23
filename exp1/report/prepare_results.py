"""从现有实验输出生成 2.1、2.2 的图表，不重新推理或改写实验数据。"""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "Depth-Anything-V2"
FIGURES = ROOT / "figures"
IDS = [f"nyu_{i:04d}" for i in (1, 101, 201, 401, 601, 801, 1001, 1201)]
SIZES = (518, 280)
KEYS = ("abs_rel", "depth_rmse_m", "xyz_rmse_m")


def metric_cell(value, best, precision=4):
    """用未舍入数值选出最小值；并列最优同时加粗。"""
    text = f"{value:.{precision}f}"
    return r"\textbf{" + text + "}" if np.isclose(value, best, rtol=1e-12, atol=1e-12) else text


def depth_grid():
    # All 16 aligned predictions share the same physical color scale.
    fig = plt.figure(figsize=(7.15, 8.50), facecolor="white")
    left, width, gap = 0.064, 0.223, 0.008
    height = width * 7.15 / 8.50 * 0.75
    tops = [0.966, 0.966-height-0.006, 0.966-2*(height+0.006),
            0.476, 0.476-height-0.006, 0.476-2*(height+0.006)]
    for block in range(2):
        for col in range(4):
            sid = IDS[4*block+col]
            for offset, label in enumerate(("RGB", "518", "280")):
                row = block*3+offset
                ax = fig.add_axes([left+col*(width+gap), tops[row]-height, width, height])
                if offset == 0:
                    ax.imshow(Image.open(DATA / "nyu_data/rgb" / f"{sid}.png"))
                    ax.set_title(sid[4:], fontsize=9, color="#203C5D", pad=3)
                else:
                    z = np.load(DATA / f"outputs/relative/{label}/aligned/{sid}_depth_m.npy")
                    im = ax.imshow(z, cmap="magma_r", vmin=0.1, vmax=10, interpolation="nearest")
                ax.set_axis_off()
                if col == 0:
                    fig.text(left-0.009, tops[row]-height/2, label, ha="right", va="center",
                             fontsize=8, color="#203C5D")
    cax = fig.add_axes([0.32, 0.018, 0.48, 0.013])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0.1, 2, 4, 6, 8, 10])
    cb.ax.tick_params(labelsize=7, length=2, pad=2)
    fig.text(0.30, 0.024, "Depth (m)", fontsize=8, ha="right", va="center", color="#526171")
    fig.savefig(FIGURES / "relative_depth_grid.png", dpi=320)
    plt.close(fig)


def pointcloud_pair():
    sid = "nyu_0001"
    clouds = [o3d.io.read_point_cloud(str(DATA / f"outputs/relative/518/aligned/{sid}_{kind}.ply"))
              for kind in ("gt", "pred")]
    # One orthographic camera, one scale and one center for both clouds; no registration.
    yaw, pitch = np.deg2rad(25), np.deg2rad(-12)
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    rotation = rx @ ry
    center = np.asarray(clouds[0].points).mean(axis=0)
    projected = [(np.asarray(cloud.points)-center) @ rotation.T for cloud in clouds]
    all_points = np.concatenate(projected)
    lo, hi = all_points[:, :2].min(axis=0), all_points[:, :2].max(axis=0)
    margin = 0.035 * (hi-lo)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.5), facecolor="white")
    fig.subplots_adjust(left=0, right=1, bottom=0.01, top=0.89, wspace=0.03)
    for ax, cloud, points, title in zip(axes, clouds, projected, ("Reference", "Depth Anything V2 / 518")):
        order = np.argsort(points[:, 2])[::-1]
        ax.scatter(points[order, 0], -points[order, 1], c=np.asarray(cloud.colors)[order],
                   s=0.17, marker=".", linewidths=0, rasterized=True)
        ax.set_xlim(lo[0]-margin[0], hi[0]+margin[0])
        ax.set_ylim(-hi[1]-margin[1], -lo[1]+margin[1])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=9, color="#203C5D", pad=4)
    fig.savefig(FIGURES / "relative_pointcloud_pair.png", dpi=320)
    plt.close(fig)


def tables():
    groups = {}
    times = {}
    for size in SIZES:
        path = DATA / f"outputs/relative/{size}/aligned/metrics.csv"
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 8 and {r["image_id"] for r in rows} == set(IDS)
        assert all(r["model"] == "relative" and r["protocol"] == "aligned" for r in rows)
        groups[size] = {r["image_id"]: np.array([float(r[k]) for k in KEYS]) for r in rows}
        times[size] = json.loads((DATA / f"outputs/relative/{size}/prediction/inference_timing.json").read_text())
        assert times[size]["image_count"] == 8 and times[size]["warmup_runs"] == 0
        assert np.isclose(np.mean([r["seconds"] for r in times[size]["images"]]), times[size]["average_seconds"])
    means = {size: np.mean(list(group.values()), axis=0) for size, group in groups.items()}
    rows = []
    for sid in IDS:
        cells = [metric_cell(groups[size][sid][i], min(groups[s][sid][i] for s in SIZES))
                 for i in range(3) for size in SIZES]
        rows.append(sid[4:] + " & " + " & ".join(cells) + r" \\")
    cells = [metric_cell(means[size][i], min(means[s][i] for s in SIZES))
             for i in range(3) for size in SIZES]
    rows.append(r"\rowcolor{ReportSummary}\textbf{均值} & " + " & ".join(cells) + r" \\")
    header = r"""\ReportTableCaption{两种输入尺寸的逐图误差及均值（对齐后；粗体为同一行、同一指标的较小值）}
\begin{ReportTable}{lCCCCCC}
\rowcolor{ReportNavy}
\ReportHead{图像编号} & \multicolumn{2}{c}{\ReportHead{AbsRel}} & \multicolumn{2}{c}{\ReportHead{深度 RMSE / m}} & \multicolumn{2}{c}{\ReportHead{XYZ RMSE / m}} \\
\rowcolor{ReportNavy}
& \ReportHead{518} & \ReportHead{280} & \ReportHead{518} & \ReportHead{280} & \ReportHead{518} & \ReportHead{280} \\
"""
    (ROOT / "relative-metric-rows.tex").write_text(header+"\n".join(rows)+"\n"+r"\end{ReportTable}"+"\n")
    timing_rows = []
    for size, shape in ((518, r"$686\times518$"), (280, r"$378\times280$")):
        t = times[size]
        average = metric_cell(1000*t["average_seconds"], min(1000*times[s]["average_seconds"] for s in SIZES), 1)
        total = metric_cell(t["total_seconds"], min(times[s]["total_seconds"] for s in SIZES), 3)
        timing_rows.append(f'{size} & {shape} & {average} & {total}' + r" \\")
    header = r"""\ReportTableCaption{两种输入尺寸的推理耗时（粗体为较短耗时）}
\begin{ReportTable}{CCCC}
\rowcolor{ReportNavy}
\ReportHead{输入尺寸} & \ReportHead{实际输入宽高} & \ReportHead{平均耗时 / ms} & \ReportHead{8 张总耗时 / s} \\
"""
    (ROOT / "relative-timing-rows.tex").write_text(header+"\n".join(timing_rows)+"\n"+r"\end{ReportTable}"+"\n")
    print("Mean error reduction 518 vs 280 (%):", (1-means[518]/means[280])*100)
    print("Time ratio 518 / 280:", times[518]["average_seconds"]/times[280]["average_seconds"])
    print("518 wins by metric:", sum((groups[518][sid] < groups[280][sid]).astype(int) for sid in IDS))


def model_comparison_table():
    """按输入尺寸列出三组均值，并与逐图结果核对。"""
    with (DATA / "outputs/summary/means.csv").open(encoding="utf-8-sig", newline="") as f:
        summary = {(r["model"], int(r["input_size"]), r["protocol"]): r for r in csv.DictReader(f)}
    lines = [r"""\ReportTableCaption{两种模型的深度与点云误差（8 张图等权均值；粗体为同尺寸下两种模型对齐后的最优值）}
\begin{ReportTable}{p{176bp}ccCCC}
\rowcolor{ReportNavy}
\ReportHead{模型} & \ReportHead{尺寸} & \ReportHead{处理} & \ReportHead{AbsRel} & \ReportHead{\shortstack{深度 RMSE\\/ m}} & \ReportHead{\shortstack{XYZ RMSE\\/ m}} \\"""]
    for size in SIZES:
        best_aligned = np.min([[float(summary[model, size, "aligned"][key+"_mean"])
                                for key in KEYS] for model in ("relative", "metric")], axis=0)
        for model, protocol in (("relative", "aligned"), ("metric", "aligned"), ("metric", "raw")):
            with (DATA / f"outputs/{model}/{size}/{protocol}/metrics.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 8 and {r["image_id"] for r in rows} == set(IDS)
            assert all(r["model"] == model and int(r["input_size"]) == size and r["protocol"] == protocol for r in rows)
            means = np.mean([[float(row[key]) for key in KEYS] for row in rows], axis=0)
            recorded = summary[model, size, protocol]
            assert int(recorded["image_count"]) == 8
            assert recorded["averaging_method"] == "equal_weight_per_image"
            assert np.allclose(means, [float(recorded[key+"_mean"]) for key in KEYS], rtol=1e-12, atol=1e-12)
            name = "Depth Anything V2" + (" (Metric Depth)" if model == "metric" else "")
            treatment = "对齐后" if protocol == "aligned" else "原始米制"
            cells = [metric_cell(value, best_aligned[i]) if protocol == "aligned" else f"{value:.4f}"
                     for i, value in enumerate(means)]
            lines.append(f"{name} & {size} & {treatment} & " + " & ".join(cells) + r" \\")
    lines.append(r"\end{ReportTable}")
    (ROOT / "model-comparison-table.tex").write_text("\n".join(lines)+"\n")
    gap_lines = [r"""\ReportTableCaption{米制深度误差超出基准的比例（原始米制 $\rightarrow$ 对齐后；粗体为较小差距）}
\begin{ReportTable}{cCCC}
\rowcolor{ReportNavy}
\ReportHead{输入尺寸} & \ReportHead{AbsRel} & \ReportHead{深度 RMSE} & \ReportHead{XYZ RMSE} \\"""]
    for size in SIZES:
        cells = []
        for key in KEYS:
            base = float(summary["relative", size, "aligned"][key+"_mean"])
            assert base > 0
            gaps = [100*(float(summary["metric", size, protocol][key+"_mean"])/base-1)
                    for protocol in ("raw", "aligned")]
            values = [metric_cell(value, min(gaps), 1) + r"\%" for value in gaps]
            cells.append(r" $\rightarrow$ ".join(values))
        gap_lines.append(str(size) + " & " + " & ".join(cells) + r" \\")
    gap_lines.append(r"\end{ReportTable}")
    (ROOT / "alignment-gap-table.tex").write_text("\n".join(gap_lines)+"\n")
    print("Model comparison: six group means verified against all 48 per-image rows.")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    depth_grid()
    pointcloud_pair()
    tables()
    model_comparison_table()
