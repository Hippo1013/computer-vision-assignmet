"""交通标志 HOG 特征提取：--params old/new 分别使用原始和调优参数。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import skimage
from skimage.feature import hog


BASE_DIR = Path(__file__).resolve().parent
CONFIGS = {
    "old": {"orientations": 9, "pixels_per_cell": (8, 8),
            "cells_per_block": (2, 2), "C": 1.0},
    "new": {"orientations": 9, "pixels_per_cell": (4, 4),
            "cells_per_block": (4, 4), "C": 0.2},
}
IMAGE_SIZE = (224, 224)
SEED = 20261022


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def preprocess(image_bytes):
    """JPEG 解码为原尺寸 RGB，再沿用原脚本的灰度化系数，不缩放。"""
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码图像")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != IMAGE_SIZE:
        raise ValueError(f"预期原图为224×224，实际为 {image.shape[:2]}；不自动缩放")
    gray = (image[:, :, 0] * 0.2989 + image[:, :, 1] * 0.5870
            + image[:, :, 2] * 0.1140)
    return gray / 255.0


def extract_split(dataset_root, split, class_names, config, output_dir):
    split_dir = dataset_root / f"{split}_dataset"
    actual_classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    if actual_classes != class_names:
        raise ValueError(f"{split}: 训练集与测试集类别不一致")
    labels, paths, samples = [], [], []
    counts = Counter()
    started = time.perf_counter()
    total = sum(len(list((split_dir / name).glob("*.jpg"))) for name in class_names)
    feature_path = output_dir / f"{split}_features.npy"
    X = None
    index = 0
    for label, class_name in enumerate(class_names):
        files = sorted((split_dir / class_name).glob("*.jpg"))
        if not files:
            raise ValueError(f"类别目录中没有 JPG：{split_dir / class_name}")
        for image_path in files:
            image_bytes = image_path.read_bytes()
            fd = hog(preprocess(image_bytes), orientations=config["orientations"],
                     pixels_per_cell=config["pixels_per_cell"],
                     cells_per_block=config["cells_per_block"],
                     block_norm="L2-Hys", transform_sqrt=False, feature_vector=True)
            relative_path = image_path.relative_to(dataset_root).as_posix()
            if not np.isfinite(fd).all():
                raise ValueError(f"HOG 特征包含非有限数值：{relative_path}")
            if X is None:
                X = np.lib.format.open_memmap(feature_path, mode="w+",
                                              dtype=np.float64, shape=(total, len(fd)))
            # 逐张写入磁盘映射矩阵，不在内存中累计整套原尺寸特征。
            X[index] = fd
            index += 1
            labels.append(label)
            paths.append(relative_path)
            samples.append({"path": relative_path, "label": label,
                            "sha256": hashlib.sha256(image_bytes).hexdigest()})
            counts[class_name] += 1
            if index % 500 == 0:
                print(f"{split}: {index}/{total} images", flush=True)
        print(f"{split}: {class_name} = {counts[class_name]}", flush=True)
    y = np.asarray(labels, dtype=np.int64)
    if X is None or index != total:
        raise ValueError(f"{split}: 图像数量不一致或没有图像")
    X.flush()
    dimension = X.shape[1]
    del X
    labels_path = output_dir / f"{split}_labels.npz"
    np.savez(labels_path, y=y, paths=np.asarray(paths))
    seconds = time.perf_counter() - started
    print(f"{split}: {len(y)} images, {dimension} dimensions, {seconds:.2f}s", flush=True)
    return {"count": len(y), "dimension": dimension, "class_counts": dict(counts),
            "feature_file": feature_path.name, "feature_sha256": sha256(feature_path),
            "labels_file": labels_path.name, "labels_sha256": sha256(labels_path),
            "extraction_seconds": seconds, "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", choices=CONFIGS, default="new")
    parser.add_argument("--dataset-root", type=Path, default=BASE_DIR / "Traffic_sign")
    parser.add_argument("--output-root", type=Path, default=BASE_DIR / "outputs/traffic_sign")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_root.resolve() / args.params
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，请使用新的 --output-root：{output_dir}")
    class_names = sorted(p.name for p in (dataset_root / "train_dataset").iterdir()
                         if p.is_dir())
    if len(class_names) < 2:
        raise ValueError("分类任务至少需要两个类别")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = CONFIGS[args.params]
    manifest = {
        "parameter_set": args.params, "dataset_root": str(dataset_root),
        "class_names": class_names,
        "class_to_label": {name: i for i, name in enumerate(class_names)},
        "hog": {key: value for key, value in config.items() if key != "C"},
        "svm_C": config["C"],
        "preprocessing": {"image_size": list(IMAGE_SIZE), "resize": "none",
                          "rgb_gray_weights": [0.2989, 0.5870, 0.1140],
                          "gray_divisor": 255.0, "block_norm": "L2-Hys",
                          "transform_sqrt": False, "dtype": "float64"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__, "scikit_image": skimage.__version__},
        "extractor_sha256": sha256(__file__), "splits": {},
    }
    for split in ("train", "test"):
        manifest["splits"][split] = extract_split(
            dataset_root, split, class_names, config, output_dir)
    if manifest["splits"]["train"]["dimension"] != manifest["splits"]["test"]["dimension"]:
        raise ValueError("训练集与测试集特征维度不同")
    sample_records = [sample for split in ("train", "test")
                      for sample in manifest["splits"][split]["samples"]]
    manifest["dataset_sha256"] = hashlib.sha256(
        json.dumps(sample_records, sort_keys=True).encode()).hexdigest()
    write_json(output_dir / "manifest.json", manifest)
    print(f"Features and manifest saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
