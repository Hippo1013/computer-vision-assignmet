#!/usr/bin/env python3
"""汇总已完成的预测、几何过滤与融合；不重新推理。"""
import csv, json
import numpy as np
from mvsnet_local.io import ROOT, save_json


def write_csv(path, rows):
    if rows:
        with open(path, "w") as f:
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)


def main():
    out = ROOT / "outputs/summary"
    out.mkdir(exist_ok=True, parents=True)
    predictions = []
    groups = []
    all_rows = []
    for views in [3, 5]:
        root = ROOT / "outputs" / f"views{views}"
        timing = root / "prediction/inference_timing.json"
        if timing.exists():
            for index, r in json.loads(timing.read_text())["per_view"].items():
                predictions.append(
                    {
                        "views": views,
                        "view": int(index),
                        **{k: v for k, v in r.items() if k != "source_ids"},
                    }
                )
        for summary in sorted(root.glob("fusion/*/summary.json")):
            groups.append(json.loads(summary.read_text()))
            for row in csv.DictReader(open(summary.with_name("metrics.csv"))):
                all_rows.append(
                    {"views": views, "prob_threshold": groups[-1]["prob_threshold"], **row}
                )
    write_csv(out / "prediction_metrics.csv", predictions)
    write_csv(out / "fusion_metrics.csv", all_rows)
    write_csv(out / "fusion_summary.csv", groups)
    means = []
    for views in [3, 5]:
        rows = [r for r in predictions if r["views"] == views]
        if rows:
            means.append(
                {
                    "views": views,
                    "reference_views": len(rows),
                    **{
                        k: float(np.mean([r[k] for r in rows]))
                        for k in ["matching_seconds", "mean_confidence", "fraction_prob_08"]
                    },
                }
            )
    write_csv(out / "prediction_means.csv", means)
    save_json(
        out / "results.json",
        {
            "prediction": means,
            "fusion": groups,
            "scope": "DTU scan9 49 reference views; no ground-truth depth or reference point cloud in supplied archive",
            "interpretation": "confidence, retained count and cross-view residual are internal consistency statistics, not accuracy metrics",
        },
    )
    print(f"汇总：{out}")


if __name__ == "__main__":
    main()
