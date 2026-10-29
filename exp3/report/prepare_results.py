"""Verify requirement 1 outputs and prepare report assets without running HOG."""
from pathlib import Path
import json
import shutil

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "cv_homework1"


def main():
    original = cv2.imread(str(SOURCE / "jinx.jpg"))
    saved_path = SOURCE / "outputs/grayscale/jinx_gray.png"
    saved = cv2.imread(str(saved_path), cv2.IMREAD_GRAYSCALE)
    if original is None or saved is None:
        raise FileNotFoundError("The source image and existing grayscale output are required")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if not np.array_equal(gray, saved):
        raise ValueError("Saved grayscale output does not match the source conversion")
    histogram = cv2.calcHist([saved], [0], None, [256], [0, 256]).ravel()
    if int(histogram.sum()) != gray.size:
        raise ValueError("Histogram pixel count mismatch")
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    shutil.copy2(SOURCE / "jinx.jpg", figures / "jinx.jpg")
    shutil.copy2(saved_path, figures / "jinx_gray.png")
    fig, ax = plt.subplots(figsize=(8.5, 2.15), layout="constrained")
    ax.bar(np.arange(256), histogram, width=1, color="#087E88")
    ax.set(xlabel="Gray level", ylabel="Pixel count", xlim=(-0.5, 255.5), ylim=(0, 9000))
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.savefig(figures / "histogram.png", dpi=240)
    plt.close(fig)
    metrics = dict(height=int(gray.shape[0]), width=int(gray.shape[1]),
                   pixel_count=int(gray.size), histogram_sum=int(histogram.sum()),
                   minimum=int(gray.min()), maximum=int(gray.max()),
                   mean=float(gray.mean()), mode=int(histogram.argmax()),
                   mode_count=int(histogram.max()), below128_percent=float((gray < 128).mean()*100),
                   saved_grayscale_matches=True)
    (figures / "grayscale_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
