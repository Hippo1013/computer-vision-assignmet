"""Plot completed full-dataset C experiments with fixed squared_hinge loss."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "outputs/svm_tuning"
    trials = json.loads((output / "trials.json").read_text())
    selected = json.loads((output / "selection.json").read_text())
    font_path = Path("/System/Library/Fonts/STHeiti Light.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    c_trials = sorted([t for t in trials if t["loss"] == "squared_hinge"], key=lambda t:t["C"])
    for field, label, color, style in [("train_accuracy", "训练准确率", "#94a3b8", "o--"),
                                        ("test_accuracy", "测试准确率", "#2563eb", "o-")]:
        ax.semilogx([t["C"] for t in c_trials], [100*t[field] for t in c_trials], style,
                        color=color, label=label)
    ax.set(title="C 参数比较（loss 固定为 squared_hinge）", xlabel="C（对数坐标）", ylabel="准确率（%）")
    ax.grid(alpha=.2)
    ax.legend()
    c_best = selected["C_stage_best"]["C"]
    best_accuracy = 100*selected["C_stage_best"]["test_accuracy"]
    ax.scatter([c_best], [best_accuracy], color="#dc2626", s=65, zorder=4)
    ax.annotate(f"C={c_best:.8g}, {best_accuracy:.2f}%", (c_best, best_accuracy),
                xytext=(-75, -35), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#dc2626"), color="#b91c1c")
    fig.suptitle("固定 HOG 的 SVM 参数比较（每组5万张训练，1万张测试）")
    fig.tight_layout()
    fig.savefig(output / "parameter_effects.png", bbox_inches="tight")
    plt.close(fig)
    print(output / "parameter_effects.png")


if __name__ == "__main__":
    main()
