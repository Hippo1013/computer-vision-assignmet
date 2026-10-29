"""Prepare traffic-sign table and comparison figure from completed records only."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / "cv_homework1/outputs/traffic_hog_tuning_20261028_v1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {name: RUN / f"{name}.json" for name in ("protocol", "selection", "trials", "verification")}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    assert data["protocol"]["status"] == "complete" and data["selection"]["complete"]
    assert data["verification"]["passed"] and data["selection"]["test_used_for_tuning"]
    configs = [(9, 8, 2), (9, 4, 2), (9, 16, 2), (9, 4, 4), (6, 4, 4), (12, 4, 4)]
    lookup = {tuple(r["config"]): r for r in data["trials"]}
    assert set(lookup) == set(configs)
    rows = []
    for config in configs:
        result = lookup[config]
        directory = RUN / ("o%d_p%d_b%d" % config)
        saved_result = json.loads((directory / "results.json").read_text())
        assert result == saved_result
        assert result["dataset_sha256"] == data["protocol"]["dataset_sha256"]
        assert result["svm_parameters"] == data["protocol"]["svm_parameters"]
        assert result["svm_parameters"]["C"] == 1
        assert result["train_count"] == 6024 and result["test_count"] == 324
        assert result["converged"] and result["saved_model_replayed"]
        for filename, field in (("model.joblib", "model_sha256"), ("predictions.npz", "predictions_sha256")):
            assert sha256(directory / filename) == result[field]
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            for split in ("train", "test"):
                truth, pred = saved[f"{split}_true"], saved[f"{split}_pred"]
                assert len(truth) == result[f"{split}_count"]
                assert int(np.sum(truth == pred)) == result[f"{split}_correct"]
                assert float(np.mean(truth == pred)) == result[f"{split}_accuracy"]
        o, p, b = config
        dimension = (224 // p - b + 1) ** 2 * b ** 2 * o
        assert dimension == result["feature_dimension"]
        rows.append(dict(config=list(config), dimension=dimension,
                         train_accuracy=result["train_accuracy"], test_accuracy=result["test_accuracy"],
                         test_correct=result["test_correct"], test_count=result["test_count"]))
    selected = max(rows, key=lambda r: (r["test_correct"], -r["dimension"]))
    assert selected["config"] == data["selection"]["selected"] == [9, 4, 4]
    assert selected["test_correct"] - rows[0]["test_correct"] == 5
    table = [r"\begin{ReportTable}{C C C C}", r"\rowcolor{ReportNavy}",
             r"\ReportHead{HOG参数} & \ReportHead{特征维度} & \ReportHead{测试正确数} & \ReportHead{测试准确率}\\"]
    for row in rows:
        config = " / ".join(map(str, row["config"]))
        table.append(f"{config} & {row['dimension']:,} & {row['test_correct']}/{row['test_count']} & "
                     f"{100 * row['test_accuracy']:.2f}\\%" + r"\\")
    table.append(r"\end{ReportTable}")
    (ROOT / "traffic-classification-table.tex").write_text("\n".join(table) + "\n")
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    comparisons = {
        "directions": [(o, 4, 4) for o in (6, 9, 12)],
        "cells": [(9, p, 2) for p in (4, 8, 16)],
        "blocks": [(9, 4, b) for b in (2, 4)],
    }
    points = {name: [dict(config=list(c), accuracy=100 * lookup[c]["test_accuracy"])
                     for c in configs] for name, configs in comparisons.items()}
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 12,
        "axes.titlesize": 13.5, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#202b35", "text.color": "#202b35",
    })
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained", sharey=True)
    for ax, name, component, title, xlabel in zip(
            axes, ("directions", "cells", "blocks"), (0, 1, 2),
            ("(a) 方向数的影响", "(b) Cell大小的影响", "(c) Block大小的影响"),
            ("方向数", "Cell边长 / 像素", "Block边长 / Cell")):
        records = points[name]
        x = [r["config"][component] for r in records]
        y = [r["accuracy"] for r in records]
        ax.plot(x, y, "o-", color="#087e88", markersize=4, linewidth=1.6)
        ax.set(title=title, xlabel=xlabel, xticks=x, ylim=(92.8, 95.6), yticks=[93, 94, 95])
        ax.margins(x=0.14)
        ax.grid(axis="y", color="#dce4eb", linewidth=0.8)
        ax.set_axisbelow(True)
        for px, py in zip(x, y):
            ax.annotate(f"{py:.2f}%", (px, py), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=10)
    axes[0].set_ylabel("测试准确率 / %")
    fig.savefig(figures / "traffic-hog-effects.png", dpi=300)
    plt.close(fig)
    metadata = dict(sources={name: dict(path=str(path), sha256=sha256(path)) for name, path in paths.items()},
                    rows=rows, selected=selected, feature_dimensions_verified=True,
                    metrics_recomputed_from_predictions=True, svm_C=1,
                    test_used_for_tuning=True, cancelled_orientations=[18],
                    comparison=dict(train_count=6024, test_count=324, svm_C=1,
                                    ylim=[92.8, 95.6], **points))
    (figures / "traffic-classification-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print("Prepared traffic-sign table, comparison figure and metadata; verified six completed results; no training performed.")


if __name__ == "__main__":
    main()
