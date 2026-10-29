"""Prepare requirement-4 figures and table from completed experiments only."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT.parent / "cv_homework1/outputs"


def main():
    sources = {
        "hog_trials": OUTPUTS / "hog_tuning/trials.json",
        "hog_final": OUTPUTS / "hog_tuning/final_evaluation.json",
        "svm_selection": OUTPUTS / "svm_tuning/selection.json",
    }
    loaded = {name: json.loads(path.read_text()) for name, path in sources.items()}
    trials = [t for t in loaded["hog_trials"] if t["phase"] == "full"]
    lookup = {tuple(t["config"]): t for t in trials}
    assert len(lookup) == len(trials)
    assert all(t["train_count"] == 40000 and t["validation_count"] == 10000
               and t["converged"] for t in trials)

    def points(configs):
        return [dict(config=list(config), accuracy=100 * lookup[config]["validation_accuracy"])
                for config in configs]

    directions = points([(o, 8, 2) for o in (4, 6, 9, 12, 15, 16, 18, 20, 21, 24, 28, 32)])
    cells = points([(9, p, 2) for p in (4, 5, 6, 8, 10, 16)])
    blocks = {p: points([(9, p, b) for b in range(1, 9 if p == 4 else 5)])
              for p in (4, 6, 8)}

    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 12,
        "axes.titlesize": 13.5, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#202b35", "text.color": "#202b35",
    })
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained", sharey=True)
    for ax in axes:
        ax.set(ylim=(38, 57), yticks=[40, 45, 50, 55])
        ax.grid(axis="y", color="#dce4eb", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].plot([r["config"][0] for r in directions], [r["accuracy"] for r in directions],
                 "o-", color="#087e88", markersize=4, linewidth=1.6)
    axes[0].set(title="(a) 方向数的影响", xlabel="方向数", ylabel="验证准确率 / %",
                xticks=[4, 9, 18, 24, 32])
    axes[1].plot([r["config"][1] for r in cells], [r["accuracy"] for r in cells],
                 "o-", color="#087e88", markersize=4, linewidth=1.6)
    axes[1].set(title="(b) Cell大小的影响", xlabel="Cell边长 / 像素", xticks=[4, 6, 8, 10, 16])
    for (p, records), color, marker in zip(blocks.items(), ["#087e88", "#203c5d", "#b67336"],
                                           ["o", "s", "^"]):
        x = [r["config"][2] for r in records]
        y = [r["accuracy"] for r in records]
        axes[2].plot(x, y, marker=marker, color=color, markersize=4, linewidth=1.6,
                     label=f"Cell {p}×{p}")
        best = max(records, key=lambda r: r["accuracy"])
        axes[2].scatter(best["config"][2], best["accuracy"], s=65,
                        facecolors="none", edgecolors=color, linewidths=1.1, zorder=4)
    axes[2].set(title="(c) Cell和Block的共同影响", xlabel="Block边长 / Cell", xticks=[1, 2, 3, 4, 6, 8])
    axes[2].legend(loc="lower right", frameon=False, fontsize=10.5)
    fig.savefig(figures / "cifar-hog-effects.png", dpi=300)
    plt.close(fig)

    final = {tuple(t["config"]): t for t in loaded["hog_final"]}
    selected = loaded["svm_selection"]["selected"]
    assert loaded["svm_selection"]["hog_config"] == [9, 4, 4]
    assert selected["C"] == 0.2 and selected["converged"]
    assert loaded["svm_selection"]["test_used_for_tuning"]
    assert all(t["train_count"] == 50000 and t["test_count"] == 10000
               for t in [*final.values(), selected])
    assert final[(9, 4, 4)]["feature_dimension"] == (32 // 4 - 4 + 1) ** 2 * 4 ** 2 * 9 == 3600
    rows = [
        ("原始参数", "9 / 8 / 2", 324, 1, final[(9, 8, 2)]),
        ("调整HOG", "9 / 4 / 4", 3600, 1, final[(9, 4, 4)]),
        ("调整HOG与SVM", "9 / 4 / 4", 3600, 0.2, selected),
    ]
    table = [r"\begin{ReportTable}{L C C C C C}", r"\rowcolor{ReportNavy}",
             r"\ReportHead{参数组合} & \ReportHead{HOG参数} & \ReportHead{特征维度} & \ReportHead{$C$} & \ReportHead{训练准确率} & \ReportHead{测试准确率}\\"]
    for label, config, dim, c_value, result in rows:
        table.append(f"{label} & {config} & {dim:,} & {c_value:g} & "
                     f"{100 * result['train_accuracy']:.2f}\\% & "
                     f"{100 * result['test_accuracy']:.2f}\\%" + r"\\")
    table.append(r"\end{ReportTable}")
    (ROOT / "cifar-classification-table.tex").write_text("\n".join(table) + "\n")
    metadata = {
        "sources": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for name, path in sources.items()},
        "comparison": {"train_count": 40000, "validation_count": 10000, "svm_C": 1,
                       "directions": directions, "cells": cells, "blocks": blocks},
        "classification_rows": [dict(label=label, hog=config, feature_dimension=dim, C=c,
                                     train_accuracy=r["train_accuracy"], test_accuracy=r["test_accuracy"])
                                for label, config, dim, c, r in rows],
        "test_used_for_svm_tuning": True,
        "feature_dimension_verified": True,
    }
    (figures / "cifar-classification-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print("Prepared requirement-4 figure, result table and metadata; no training performed.")


if __name__ == "__main__":
    main()
