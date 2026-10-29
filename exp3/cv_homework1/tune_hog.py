"""Controlled CIFAR-10 HOG search; reuse extractFeat's grayscale/HOG implementation."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import warnings

import joblib
from joblib import Parallel, delayed
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
import skimage

from extractFeat import hog, rgb2gray, unpickle

ROOT = Path(__file__).resolve().parent
BASELINE = (9, 8, 2)
SEED = 20261022
SVM = dict(C=1.0, max_iter=1000, tol=1e-4, dual="auto", random_state=SEED)


def key(config):
    return "o%d_p%d_b%d" % tuple(config)


def parameters(config):
    o, p, b = config
    return dict(orientations=o, pixels_per_cell=(p, p), cells_per_block=(b, b))


def dimension(config):
    o, p, b = config
    return (32 // p - b + 1) ** 2 * b ** 2 * o


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    partial.replace(path)


def load_images(data_dir, test=False):
    paths = [data_dir / "test_batch"] if test else [data_dir / f"data_batch_{i}" for i in range(1, 6)]
    batches = [unpickle(path) for path in paths]
    raw = np.concatenate([batch[b"data"] for batch in batches])
    labels = np.concatenate([np.asarray(batch[b"labels"], dtype=np.int64) for batch in batches])
    names = [name.decode() for batch in batches for name in batch[b"filenames"]]
    # Exactly the channel order, grayscale coefficients and float64 arithmetic of getFeat.
    rgb = raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    gray = rgb2gray(rgb.transpose(1, 2, 3, 0)).transpose(2, 0, 1) / 255.0
    expected = 10000 if test else 50000
    if gray.shape != (expected, 32, 32) or not np.array_equal(np.bincount(labels), np.full(10, expected // 10)):
        raise ValueError("Unexpected CIFAR-10 size or class distribution")
    return gray, labels, names


def chunk_features(images, config):
    return np.asarray([hog(image, **parameters(config)) for image in images])


def extract(images, config, workers, scratch):
    start = time.perf_counter()
    path = scratch / (key(config) + ".npy")
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64,
                                       shape=(len(images), dimension(config)))
    chunks = Parallel(n_jobs=workers, return_as="generator", temp_folder=str(scratch),
                      inner_max_num_threads=1)(
        delayed(chunk_features)(images[i:i + 500], config) for i in range(0, len(images), 500))
    offset = 0
    for chunk in chunks:
        matrix[offset:offset + len(chunk)] = chunk
        offset += len(chunk)
    if offset != len(images) or not np.isfinite(matrix).all():
        raise ValueError("Incomplete or non-finite HOG features")
    matrix.flush()
    return matrix, time.perf_counter() - start, path


def fit_score(features, labels, train_idx, validation_idx):
    clf = LinearSVC(**SVM)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(features[train_idx], labels[train_idx])
        fit_seconds = time.perf_counter() - start
        train_acc = accuracy_score(labels[train_idx], clf.predict(features[train_idx]))
        predictions = clf.predict(features[validation_idx])
    return clf, dict(train_accuracy=float(train_acc), validation_accuracy=float(
        accuracy_score(labels[validation_idx], predictions)), fit_seconds=fit_seconds,
        n_iter=int(np.max(clf.n_iter_)), converged=not any(
            issubclass(w.category, ConvergenceWarning) for w in caught),
        warnings=[str(w.message) for w in caught])


def run_search(args, output, scratch):
    gray, labels, names = load_images(args.data_dir)
    split_path = output / "splits.npz"
    if split_path.exists():
        splits = np.load(split_path)
        train_idx, val_idx, coarse_idx = (splits[name] for name in ("train", "validation", "coarse"))
    else:
        train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=10000,
                                              stratify=labels, random_state=SEED)
        coarse_idx, _ = train_test_split(train_idx, train_size=10000,
                                         stratify=labels[train_idx], random_state=SEED)
        np.savez(split_path, train=train_idx, validation=val_idx, coarse=coarse_idx)
    if set(train_idx) & set(val_idx) or not set(coarse_idx) <= set(train_idx):
        raise AssertionError("Invalid training/validation split")
    results_path = output / "trials.json"
    trials = json.loads(results_path.read_text()) if results_path.exists() else []
    config_list = [tuple(map(int, item.split(","))) for item in args.configs]
    if args.phase == "single":
        config_list = list(dict.fromkeys([BASELINE]
            + [(o, 8, 2) for o in [4, 6, 12, 18]]
            + [(9, p, 2) for p in [4, 6, 10, 16]]
            + [(9, 8, b) for b in [1, 3, 4]]))
    if not config_list:
        raise ValueError("Provide --configs o,p,b ...")
    for config in config_list:
        if min(config) < 1 or config[2] > 32 // config[1]:
            raise ValueError(f"Invalid HOG config: {config}")
        if any(tuple(t["config"]) == config and t["phase"] == args.phase for t in trials):
            print("SKIP", args.phase, key(config), flush=True)
            continue
        idx = train_idx if args.phase == "full" else coarse_idx
        # Only selected training rows and the fixed validation set are extracted.
        selected = np.concatenate((idx, val_idx))
        print("START", args.phase, key(config), "dimension", dimension(config), flush=True)
        matrix, seconds, cache = extract(gray[selected], config, args.workers, scratch)
        clf, metrics = fit_score(matrix, labels[selected], np.arange(len(idx)),
                                 np.arange(len(idx), len(selected)))
        result = dict(phase=args.phase, config=config, parameters=parameters(config),
                      feature_dimension=matrix.shape[1], train_count=len(idx),
                      validation_count=len(val_idx), extraction_seconds=seconds, **metrics)
        trials.append(result)
        save_json(results_path, trials)
        if args.phase == "full":
            np.save(output / (key(config) + "_validation_predictions.npy"),
                    clf.predict(matrix[len(idx):]))
        print("RESULT", json.dumps(result), flush=True)
        del matrix, clf
        cache.unlink()
    # Check exact agreement with original extraction and an existing baseline .feat file.
    single = gray[0]
    raw = unpickle(args.data_dir / "data_batch_1")[b"data"][0]
    original_gray = rgb2gray(np.reshape(raw.reshape(3, 1024).T, (32, 32, 3))) / 255.0
    np.testing.assert_array_equal(single, original_gray)
    old_path = ROOT / "data/features/train" / (Path(names[0]).stem + ".feat")
    parity = dict(grayscale_exact_match=True, original_feature_file=str(old_path),
                  existing_feature_checked=False)
    if old_path.exists():
        old = joblib.load(old_path)
        np.testing.assert_array_equal(old[:-1], hog(single, **parameters(BASELINE)))
        assert int(old[-1]) == int(labels[0])
        parity["existing_feature_checked"] = True
    save_json(output / "extraction_parity.json", parity)


def final_evaluation(args, output, scratch):
    trials = json.loads((output / "trials.json").read_text())
    full = [t for t in trials if t["phase"] == "full" and t["converged"]]
    if not full:
        raise ValueError("No converged full-training validation candidates")
    best = max(full, key=lambda t: (t["validation_accuracy"], -t["feature_dimension"]))
    config = tuple(best["config"])
    save_json(output / "selection.json", dict(selection_metric="validation_accuracy",
        selected_config=config, selected_validation_trial=best, test_used_for_selection=False))
    gray, labels, _ = load_images(args.data_dir)
    test_gray, test_labels, _ = load_images(args.data_dir, test=True)
    results = []
    for candidate in dict.fromkeys([BASELINE, config]):
        print("FINAL START", key(candidate), flush=True)
        matrix, seconds, cache = extract(np.concatenate([gray, test_gray]), candidate,
                                         args.workers, scratch)
        clf, metrics = fit_score(matrix, np.concatenate([labels, test_labels]),
                                 np.arange(50000), np.arange(50000, 60000))
        metrics["test_accuracy"] = metrics.pop("validation_accuracy")
        if not metrics["converged"]:
            raise RuntimeError("Final SVM did not converge; inspect before reporting")
        predictions = clf.predict(matrix[50000:])
        np.save(output / (key(candidate) + "_test_predictions.npy"), predictions)
        joblib.dump(clf, output / (key(candidate) + "_model.joblib"))
        result = dict(config=candidate, parameters=parameters(candidate),
            feature_dimension=dimension(candidate), train_count=50000, test_count=10000,
            extraction_seconds=seconds, **metrics,
            classification_report=classification_report(test_labels, predictions, output_dict=True),
            confusion_matrix=confusion_matrix(test_labels, predictions).tolist())
        results.append(result)
        save_json(output / "final_evaluation.json", results)
        print("FINAL RESULT", key(candidate), metrics, flush=True)
        if candidate == config:
            # Durable final features enable later classification without extracting again.
            shutil.copyfile(cache, output / "best_features.npy")
            np.savez(output / "best_labels.npz", train=labels, test=test_labels)
        del matrix, clf
        cache.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["single", "joint", "refine", "full", "final"], required=True)
    parser.add_argument("--configs", nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/hog_tuning")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "cifar-10-batches-py")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scratch-parent", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(seed=SEED, svm=SVM, preprocessing="extractFeat.rgb2gray / 255.0; float64",
        hog_fixed=dict(block_norm="L2-Hys", transform_sqrt=False, feature_vector=True, channel_axis=None),
        python=sys.executable, python_version=sys.version, platform=platform.platform(),
        versions=dict(numpy=np.__version__, sklearn=sklearn.__version__, skimage=skimage.__version__),
        data_sha256={p.name: sha256(p) for p in sorted(args.data_dir.glob("*batch*")) if p.is_file()},
        extractor_sha256=sha256(ROOT / "extractFeat.py"), runner_sha256=sha256(Path(__file__)),
        train_count=40000, validation_count=10000, coarse_train_count=10000,
        test_policy="Only baseline and validation-selected best; 50k train / 10k test, after selection.")
    provenance_path = output / "protocol.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        for field in ["seed", "svm", "preprocessing", "hog_fixed", "versions", "data_sha256", "extractor_sha256"]:
            if previous[field] != provenance[field]:
                raise ValueError(f"Protocol changed: {field}; use a new output directory")
    else:
        save_json(provenance_path, provenance)
    scratch = Path(tempfile.mkdtemp(prefix="exp3-hog-", dir=args.scratch_parent))
    try:
        if args.phase == "final":
            final_evaluation(args, output, scratch)
        else:
            run_search(args, output, scratch)
    finally:
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
