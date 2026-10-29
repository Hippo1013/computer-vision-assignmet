"""Prepare the requirement-2 report figure from preserved, verified HOG arrays."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT.parent / "cv_homework1/outputs/hog_parameter_review_20261028_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "files.json").read_text())

    def verified_json(relative):
        path = run / relative
        assert sha(path) == manifest[relative]["sha256"], relative
        return json.loads(path.read_text())

    assert verified_json("verification.json")["passed"]
    protocol = verified_json("provenance.json")
    records = verified_json("results.json")["configurations"]
    # Check actual complete feature lengths for all seven table rows.
    for record in records:
        relative = record["id"] + "/features.npy"
        path = run / relative
        assert sha(path) == manifest[relative]["sha256"], relative
        features = np.load(path, mmap_mode="r", allow_pickle=False)
        assert features.ndim == 1 and features.size == record["feature_dimension"]
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial Unicode MS", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10.5,
                         "axes.titlesize": 12, "axes.labelsize": 10.5,
                         "xtick.labelsize": 9.5, "ytick.labelsize": 9.5})
    fig, axes = plt.subplots(3, 3, figsize=(12, 6.3), layout="constrained",
                             gridspec_kw={"height_ratios": [1, 1.65, 0.55]})
    y0, y1, x0, x1 = protocol["roi_y0_y1_x0_x1"]
    target_y, target_x = protocol["target_pixel_yx"]
    histograms = {}
    for p in (4, 8, 16):
        relative = f"o9_p{p}_b2/cell_histograms.npy"
        path = run / relative
        assert sha(path) == manifest[relative]["sha256"], relative
        histogram = np.load(path, allow_pickle=False)
        assert histogram.shape[-1] == 9
        histograms[p] = histogram[target_y // p, target_x // p].copy()
    histogram_ymax = max(values.max() for values in histograms.values()) * 1.12
    for col, p in enumerate((4, 8, 16)):
        relative = f"o9_p{p}_b2/hog_vis.npy"
        path = run / relative
        assert sha(path) == manifest[relative]["sha256"], relative
        vis = np.load(path, allow_pickle=False)
        image = np.sqrt(vis)
        kwargs = dict(cmap="gray", vmin=0, vmax=protocol["display"]["vmax"], interpolation="nearest")
        axes[0, col].imshow(image, **kwargs)
        axes[0, col].add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                            fill=False, edgecolor="#ffb454", linewidth=1.5))
        axes[0, col].set_title(f"Cell = {p} × {p} 像素", fontsize=12)
        axes[1, col].imshow(image[y0:y1, x0:x1], extent=(x0, x1, y1, y0), **kwargs)
        cell_y, cell_x = target_y // p * p, target_x // p * p
        axes[1, col].add_patch(plt.Rectangle((cell_x, cell_y), p, p,
                                            fill=False, edgecolor="#4dd7d0", linewidth=1.5))
        for ax in axes[:2, col]:
            ax.axis("off")
        axes[2, col].bar(np.arange(10, 180, 20), histograms[p], width=17, color="#087e88")
        axes[2, col].set(xlim=(0, 180), ylim=(0, histogram_ymax),
                         xlabel="梯度方向 / °", ylabel="幅值加权统计值")
        axes[2, col].set_xticks([0, 60, 120, 180])
        axes[2, col].set_yticks([0, 2, 4])
        axes[2, col].grid(axis="y", alpha=0.2)
        axes[2, col].set_axisbelow(True)
    fig.savefig(figures / "hog-cell-comparison.png", dpi=300)
    plt.close(fig)

    # Same physical ROI and display scale for each orientation count.
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), layout="constrained",
                             gridspec_kw={"height_ratios": [1, 1.65]})
    for col, orientations in enumerate((4, 9, 18)):
        relative = f"o{orientations}_p8_b2/hog_vis.npy"
        path = run / relative
        assert sha(path) == manifest[relative]["sha256"], relative
        image = np.sqrt(np.load(path, allow_pickle=False))
        kwargs = dict(cmap="gray", vmin=0, vmax=protocol["display"]["vmax"], interpolation="nearest")
        axes[0, col].imshow(image, **kwargs)
        axes[0, col].add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                            fill=False, edgecolor="#ffb454", linewidth=1.5))
        axes[0, col].set_title(f"方向数 = {orientations}", fontsize=12)
        axes[1, col].imshow(image[y0:y1, x0:x1], **kwargs)
        for ax in axes[:, col]:
            ax.axis("off")
    fig.savefig(figures / "hog-orientation-comparison.png", dpi=300)
    plt.close(fig)

    # Same physical Cell, at the [0,0] position in each enclosing Block.
    normalized_histograms = {}
    for b in (1, 2, 4):
        record = next(r for r in records if r["id"] == f"o9_p8_b{b}")
        features = np.load(run / record["id"] / "features.npy", mmap_mode="r", allow_pickle=False)
        blocks = features.reshape(record["block_shape"])
        values = np.array(blocks[target_y//8, target_x//8, 0, 0, :])
        relative = f"o9_p8_b{b}/target_cell.npz"
        assert sha(run / relative) == manifest[relative]["sha256"], relative
        with np.load(run / relative, allow_pickle=False) as saved:
            assert np.array_equal(values, saved["normalized_in_anchored_block"])
        normalized_histograms[b] = values
    normalized_ymax = max(values.max() for values in normalized_histograms.values()) * 1.12
    fig, axes = plt.subplots(1, 3, figsize=(12, 1.8), layout="constrained")
    for ax, b in zip(axes, (1, 2, 4)):
        ax.bar(np.arange(10, 180, 20), normalized_histograms[b], width=17, color="#5368b5")
        ax.set(title=f"Block = {b} × {b} Cell", xlabel="梯度方向 / °", ylabel="归一化特征值",
               xlim=(0, 180), ylim=(0, normalized_ymax))
        ax.set_xticks([0, 60, 120, 180])
        ax.set_yticks([0, 0.2, 0.4])
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(figures / "hog-block-cell-histograms.png", dpi=300)
    plt.close(fig)
    metadata = {"run": str(run), "source_results_sha256": sha(run / "results.json"),
                "source_protocol_sha256": sha(run / "provenance.json"),
                "display": protocol["display"], "roi_y0_y1_x0_x1": [y0, y1, x0, x1],
                "target_pixel_yx": [target_y, target_x],
                "histogram_stage": "Before Block normalization; no additional normalization or display transform",
                "histogram_ymax": float(histogram_ymax),
                "histograms": {str(p): {"values": values.tolist(),
                               "cell_pixel_box_yxyx": [target_y//p*p, target_x//p*p,
                                                       (target_y//p+1)*p, (target_x//p+1)*p]}
                               for p, values in histograms.items()},
                "dimensions": {r["id"]: r["feature_dimension"] for r in records},
                "feature_lengths_verified": True,
                "block_normalized_histograms": {str(b): values.tolist()
                                                 for b, values in normalized_histograms.items()},
                "block_normalized_ymax": float(normalized_ymax),
                "block_context": "Same 8x8-pixel Cell, anchored as top-left Cell in each enclosing Block",
                "block_histograms_match_complete_features": True,
                "layout": {"cell_figure_inches": [12, 6.3], "cell_row_height_ratios": [1, 1.65, 0.55],
                           "orientation_figure_inches": [12, 5.6], "orientation_row_height_ratios": [1, 1.65],
                           "block_figure_inches": [12, 1.8],
                           "plot_font_points": {"title": 12, "axis_label": 10.5, "tick_label": 9.5}},
                "figures": ["hog-cell-comparison.png", "hog-orientation-comparison.png",
                            "hog-block-cell-histograms.png"]}
    (figures / "hog-comparison-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
