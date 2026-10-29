"""Merge independent search shards and export measured HOG comparisons."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from tune_hog import BASELINE, ROOT, key, save_json, sha256


def merge(output):
    protocol = json.loads((output / "protocol.json").read_text())
    trials = json.loads((output / "trials.json").read_text())
    indexed = {(t["phase"], tuple(t["config"])): t for t in trials}
    shards = []
    for path in sorted((output / "shards").glob("*/trials.json")):
        shard_protocol = json.loads((path.parent / "protocol.json").read_text())
        for field in ["seed", "svm", "preprocessing", "hog_fixed", "versions", "data_sha256", "extractor_sha256"]:
            if protocol[field] != shard_protocol[field]:
                raise ValueError(f"Incompatible shard {path}: {field}")
        with np.load(output / "splits.npz") as master, np.load(path.parent / "splits.npz") as other:
            for name in master.files:
                np.testing.assert_array_equal(master[name], other[name])
        for trial in json.loads(path.read_text()):
            identity = (trial["phase"], tuple(trial["config"]))
            if identity in indexed and indexed[identity] != trial:
                raise ValueError(f"Conflicting duplicate trial: {identity}")
            indexed[identity] = trial
        shards.append(dict(path=str(path.relative_to(output)), sha256=sha256(path)))
    trials = list(indexed.values())
    save_json(output / "trials.json", trials)
    save_json(output / "merged_shards.json", shards)
    return trials


def table(trials):
    lines = ["|方向数|cell尺寸|block尺寸|维度|训练准确率|验证准确率|",
             "|---:|:---:|:---:|---:|---:|---:|"]
    for t in trials:
        o, p, b = t["config"]
        lines.append(f"|{o}|{p}×{p}|{b}×{b}|{t['feature_dimension']}|"
                     f"{t['train_accuracy']:.2%}|{t['validation_accuracy']:.2%}|")
    return lines


def summarize(output, trials):
    fields = ["phase", "orientations", "pixels_per_cell", "cells_per_block", "feature_dimension",
              "train_count", "validation_count", "train_accuracy", "validation_accuracy",
              "fit_seconds", "extraction_seconds", "n_iter", "converged"]
    with (output / "trials.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trial in trials:
            row = {field: trial[field] for field in fields if field in trial}
            row.update(zip(fields[1:4], trial["config"]))
            writer.writerow(row)
    full = sorted([t for t in trials if t["phase"] == "full"],
                  key=lambda t: (-t["validation_accuracy"], t["feature_dimension"]))
    single = [t for t in trials if t["phase"] == "single"]
    final_path = output / "final_evaluation.json"
    final = json.loads(final_path.read_text()) if final_path.exists() else []
    supplemental = [json.loads(p.read_text()) for p in sorted((output / "runners_up").glob("rank*_result.json"))]
    lines = ["# CIFAR-10 HOG 参数实验记录", "", "## 最终评估", ""]
    if final:
        lines += ["|配置|方向数|cell尺寸|block尺寸|维度|训练准确率|测试准确率|",
                  "|:---|---:|:---:|:---:|---:|---:|---:|"]
        for t in final:
            o, p, b = t["config"]
            name = "基线" if tuple(t["config"]) == BASELINE else "验证集最佳候选"
            lines.append(f"|{name}|{o}|{p}×{p}|{b}×{b}|{t['feature_dimension']}|"
                         f"{t['train_accuracy']:.2%}|{t['test_accuracy']:.2%}|")
        if len(final) == 2:
            lines += ["", f"测试准确率提高 {(final[1]['test_accuracy']-final[0]['test_accuracy'])*100:.2f} 个百分点。"]
    else:
        lines.append("尚未执行最终测试评估。")
    if supplemental and final:
        selected = next(t for t in final if t["config"] == full[0]["config"])
        comparison = [dict(validation_rank=1, config=selected["config"],
                           validation_accuracy_40k=full[0]["validation_accuracy"],
                           feature_dimension=selected["feature_dimension"], train_count=50000, test_count=10000,
                           train_accuracy=selected["train_accuracy"], test_accuracy=selected["test_accuracy"])]
        comparison += [{k: t[k] for k in comparison[0]} for t in supplemental]
        comparison.sort(key=lambda t: t["validation_rank"])
        save_json(output / "runners_up/comparison.json", comparison)
        lines += ["", "## 验证前三名的全量训练比较", "",
                  "各组均使用官方50000张训练图像重训，在官方另外10000张测试图像上评估。原验证准确率来自40000张训练、10000张验证的参数比较阶段。排名沿用原验证排序。",
                  "", "|原验证排名|方向数|cell尺寸|block尺寸|原验证准确率|5万张训练准确率|最终测试准确率|",
                  "|---:|---:|:---:|:---:|---:|---:|---:|"]
        for t in comparison:
            o, p, b = t["config"]
            lines.append(f"|{t['validation_rank']}|{o}|{p}×{p}|{b}×{b}|{t['validation_accuracy_40k']:.2%}|"
                         f"{t['train_accuracy']:.2%}|{t['test_accuracy']:.2%}|")
        lines += ["", "补充实验的模型、逐张预测、指标和协议保存在 `runners_up/`。原 `selection.json` 和首次 `final_evaluation.json` 保留原记录。"]
    lines += ["", "## 实验设置", "",
        "只改变 HOG 的三个参数；cell 和 block 均为正方形。沿用原有灰度化和 HOG 提取逻辑。",
        "分类器固定为 LinearSVC(C=1, max_iter=1000, tol=1e-4, dual=auto)。不做颜色特征、数据增强或特征缩放。",
        "耗时为本机墙钟时间，运行期间存在并行任务，不作为严格的速度排名。",
        "官方训练集固定分层划分40000张训练、10000张验证；当前所有候选均直接使用40000张训练图像比较，不做小样本筛选。早期12次10000张训练实验只保留为补充记录，不用于淘汰候选。",
        "最终参数只依据验证集选择，随后以50000张训练图像重训，评估10000张官方测试图像。",
        "", f"参数比较阶段共完成 {len(trials)} 次训练，覆盖 {len({tuple(t['config']) for t in trials})} 组不同参数；其中 {len(full)} 组完成40000张训练图像的复核。",
        f"最终评估另完成 {len(final)} 次50000张训练图像的重训，分别对应基线和已选参数。" if final else "最终评估尚未完成。",
        f"随后另完成 {len(supplemental)} 次50000张训练图像的次佳参数补充实验。" if supplemental else "",
        f"全部训练收敛：{all(t['converged'] for t in trials)}。最优仅指当前搜索范围内的验证集最优，不代表所有合法参数的全局最优。",
        "", "## 完整候选比较", "",
        "以下使用相同40000张训练图像和10000张验证图像，按验证准确率排序。", ""]
    lines += table(full)
    if full:
        lines += ["", "## 参数影响分析", ""]
        for changed, fixed, name in [(0, BASELINE, "方向数"), (1, BASELINE, "cell边长"),
                                     (2, BASELINE, "block边长"), (2, (9, 6, 2), "6×6 cell下的block边长")]:
            compared = sorted([t for t in full if all(t["config"][j] == fixed[j]
                              for j in range(3) if j != changed)], key=lambda t: t["config"][changed])
            if compared:
                values = "、".join(f"{t['config'][changed]}：{t['validation_accuracy']:.2%}" for t in compared)
                lines.append(f"- {name}对比（其余两项固定为{fixed}的对应值）：{values}。")
        baseline = next((t for t in full if tuple(t["config"]) == BASELINE), None)
        if baseline:
            lines += ["", f"搜索最佳组合为 {full[0]['config']}，验证准确率 {full[0]['validation_accuracy']:.2%}，"
                      f"比基线提高 {(full[0]['validation_accuracy']-baseline['validation_accuracy'])*100:.2f} 个百分点。",
                      "参数变化会同时改变特征的空间粒度、上下文范围和维度；训练准确率提高不能直接解释为分类能力提高。以上是当前固定分类器与数据划分下的实测比较。"]
    lines += ["", "## 早期补充实验", "", "以下统一使用10000张训练图像和10000张验证图像，仅为早期补充记录，不用于当前参数筛选。", ""] + table(single)
    lines += ["", "## 维度计算", "",
        "32×32输入、方向数o、cell边长p、block边长b时：D=(floor(32/p)-b+1)²×b²×o。",
        "p不能整除32时，边缘不足一个cell的像素不构成完整cell。所有实际特征维度均与该式一致。",
        "", "## 实验文件", "",
        "`protocol.json`：环境、固定设置和数据哈希；`splits.npz`：固定划分；`trials.json/csv`：逐组实测数据。",
        "`selection.json`：测试前锁定的验证集最佳候选；`final_evaluation.json`：最终训练、测试、分类别指标和混淆矩阵。",
        "`shards/` 与 `merged_shards.json`：分任务结果及合并核验记录；每份任务使用完全相同的划分和分类器。",
        "`parameter_effects.png`、`refinement_effects.png`、`search_landscape.png`：单参数对比、组合细化和完整候选的验证准确率。",
        "`*_model.joblib`、`*_test_predictions.npy`：最终模型和测试预测；`best_features.npy`：最终最佳特征。", ""]
    (output / "summary.md").write_text("\n".join(lines))
    return full


def figures(output, trials, full):
    font_path = Path("/System/Library/Fonts/STHeiti Light.ttc")
    if font_path.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for i, title in enumerate(["方向数", "cell边长（像素）", "block边长（单元）"]):
        subset = sorted([t for t in full if all(t["config"][j] == BASELINE[j]
                         for j in range(3) if j != i)], key=lambda t: t["config"][i])
        x = [t["config"][i] for t in subset]
        axes[i].plot(x, [t["train_accuracy"] * 100 for t in subset], "o--", label="训练（4万张）", color="#9ca3af")
        axes[i].plot(x, [t["validation_accuracy"] * 100 for t in subset], "o-", label="验证（1万张）", color="#2563eb")
        axes[i].set(xlabel=title, ylabel="准确率（%）")
        if i == 2:
            axes[i].set_xticks(x)
        axes[i].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("HOG 单参数影响（其余参数固定为 9 / 8 / 2）")
    fig.tight_layout()
    fig.savefig(output / "parameter_effects.png", bbox_inches="tight")
    plt.close(fig)
    if not full:
        return
    blocks = sorted({t["config"][2] for t in full})
    cells = sorted({t["config"][1] for t in full})
    orientations = sorted({t["config"][0] for t in full})
    columns = min(4, len(blocks))
    rows = math.ceil(len(blocks)/columns)
    fig, axes = plt.subplots(rows, columns, figsize=(max(10, 3.8*columns), rows*max(4, .38*len(orientations)+1.5)), squeeze=False)
    low, high = min(t["validation_accuracy"]*100 for t in full), max(t["validation_accuracy"]*100 for t in full)
    for ax, b in zip(axes.ravel(), blocks):
        values = np.full((len(orientations), len(cells)), np.nan)
        for t in full:
            o, p, block = t["config"]
            if block == b:
                values[orientations.index(o), cells.index(p)] = 100*t["validation_accuracy"]
        ax.imshow(values, vmin=low, vmax=high, cmap="YlGnBu", aspect="auto")
        ax.set(xticks=range(len(cells)), xticklabels=cells, yticks=range(len(orientations)),
               yticklabels=orientations, xlabel="cell边长（像素）", ylabel="方向数", title=f"block {b}×{b}")
        for y, x in zip(*np.where(np.isfinite(values))):
            ax.text(x, y, f"{values[y,x]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if values[y,x] > (low+high)/2 else "black")
    for ax in axes.ravel()[len(blocks):]:
        ax.axis("off")
    fig.suptitle("HOG 组合搜索验证准确率（训练4万张，验证1万张；空白为未运行）")
    fig.tight_layout()
    fig.savefig(output / "search_landscape.png", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    regions = [([t for t in full if t["config"][1:] == [5, 3]], 0,
                "方向数比较（cell 5×5，block 3×3）", "方向数"),
               ([t for t in full if t["config"][:2] == [9, 4]], 2,
                "block尺寸比较（方向数9，cell 4×4）", "block边长（单元）")]
    for ax, (subset, index, title, xlabel) in zip(axes, regions):
        subset = sorted(subset, key=lambda t: t["config"][index])
        x = [t["config"][index] for t in subset]
        ax.plot(x, [100*t["train_accuracy"] for t in subset], "o--", color="#9ca3af", label="训练")
        ax.plot(x, [100*t["validation_accuracy"] for t in subset], "o-", color="#2563eb", label="验证")
        ax.set(title=title, xlabel=xlabel, ylabel="准确率（%）", xticks=x)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for b, color in [(2, "#64748b"), (3, "#2563eb"), (4, "#d97706")]:
        subset = sorted([t for t in full if t["config"][0] == 9 and t["config"][2] == b],
                        key=lambda t: t["config"][1])
        axes[2].plot([t["config"][1] for t in subset], [100*t["validation_accuracy"] for t in subset],
                     "o-", color=color, label=f"block {b}×{b}")
    axes[2].set(title="cell与block组合比较（方向数9）", xlabel="cell边长（像素）", ylabel="验证准确率（%）")
    axes[2].grid(alpha=.2)
    axes[2].legend(fontsize=8)
    fig.suptitle("HOG 组合搜索与参数细化（训练4万张，固定验证1万张）")
    fig.tight_layout()
    fig.savefig(output / "refinement_effects.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/hog_tuning")
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    trials = merge(args.output) if args.merge_shards else json.loads((args.output / "trials.json").read_text())
    full = summarize(args.output, trials)
    figures(args.output, trials, full)
    print("Trials:", len(trials), "full:", len(full))
    for t in full[:10]:
        print(key(t["config"]), f"train={t['train_accuracy']:.4f}", f"validation={t['validation_accuracy']:.4f}")


if __name__ == "__main__":
    main()
