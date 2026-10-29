"""使用交通标志 HOG 特征训练和测试线性 SVM，并保留模型和逐张预测。"""

import argparse
import csv
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC

from extractFeat_TrafficSign import BASE_DIR, CONFIGS, IMAGE_SIZE, SEED, sha256, write_json


def load_split(output_dir, manifest, split):
    info = manifest["splits"][split]
    feature_path = output_dir / info["feature_file"]
    if sha256(feature_path) != info["feature_sha256"]:
        raise ValueError(f"{split}: 特征文件哈希不匹配")
    labels_path = output_dir / info["labels_file"]
    if sha256(labels_path) != info["labels_sha256"]:
        raise ValueError(f"{split}: 标签文件哈希不匹配")
    X = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    with np.load(labels_path, allow_pickle=False) as data:
        y, paths = data["y"], data["paths"]
    samples = info["samples"]
    if (X.shape != (info["count"], info["dimension"])
            or y.shape != (info["count"],)
            or paths.tolist() != [sample["path"] for sample in samples]
            or y.tolist() != [sample["label"] for sample in samples]):
        raise ValueError(f"{split}: 特征、标签或图像清单不一致")
    for start in range(0, len(y), 64):
        if not np.isfinite(X[start:start + 64]).all():
            raise ValueError(f"{split}: HOG 特征包含非有限数值")
    return X, y, paths


def predict_batches(clf, X):
    return np.concatenate([clf.predict(X[start:start + 64])
                           for start in range(0, len(X), 64)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", choices=CONFIGS, default="new")
    parser.add_argument("--output-root", type=Path, default=BASE_DIR / "outputs/traffic_sign")
    args = parser.parse_args()
    output_dir = args.output_root.resolve() / args.params
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = CONFIGS[args.params]
    expected_hog = {key: list(value) if isinstance(value, tuple) else value
                    for key, value in config.items() if key != "C"}
    if (manifest["parameter_set"] != args.params or manifest["hog"] != expected_hog
            or manifest["svm_C"] != config["C"]
            or manifest["preprocessing"]["image_size"] != list(IMAGE_SIZE)
            or manifest["preprocessing"]["resize"] != "none"):
        raise ValueError("特征参数与所选 old/new 参数组不一致")
    started = time.perf_counter()
    X_train, y_train, train_paths = load_split(output_dir, manifest, "train")
    X_test, y_test, test_paths = load_split(output_dir, manifest, "test")
    parameters = {"C": config["C"], "max_iter": 1000, "tol": 1e-4,
                  "dual": "auto", "loss": "squared_hinge", "penalty": "l2",
                  "random_state": SEED}
    clf = LinearSVC(**parameters)
    print(f"Training {args.params}: {len(y_train)} images, {X_train.shape[1]} dimensions, "
          f"C={config['C']}", flush=True)
    fit_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_started
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in captured)
    if not converged:
        raise RuntimeError("SVM 在原定1000次迭代内未收敛，本次不保存为完成结果")
    train_prediction = predict_batches(clf, X_train)
    test_prediction = predict_batches(clf, X_test)
    train_accuracy = accuracy_score(y_train, train_prediction)
    test_accuracy = accuracy_score(y_test, test_prediction)
    class_names = manifest["class_names"]
    labels = list(range(len(class_names)))
    report = classification_report(y_test, test_prediction, labels=labels,
                                   target_names=class_names, output_dict=True,
                                   zero_division=0)
    matrix = confusion_matrix(y_test, test_prediction, labels=labels)
    model_path = output_dir / "model.joblib"
    joblib.dump(clf, model_path)
    # 重新加载最终模型，核对训练和测试的逐张预测。
    restored = joblib.load(model_path)
    if not (np.array_equal(predict_batches(restored, X_train), train_prediction)
            and np.array_equal(predict_batches(restored, X_test), test_prediction)):
        raise RuntimeError("保存模型重放预测不一致")
    prediction_path = output_dir / "predictions.npz"
    np.savez(prediction_path, train_true=y_train, train_pred=train_prediction,
             train_paths=train_paths, test_true=y_test, test_pred=test_prediction,
             test_paths=test_paths)
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "true_class", "predicted_label",
                         "predicted_class", "correct"])
        for path, true, pred in zip(test_paths, y_test, test_prediction):
            writer.writerow([path, int(true), class_names[true], int(pred),
                             class_names[pred], int(true == pred)])
    result = {
        "parameter_set": args.params, "hog": manifest["hog"],
        "preprocessing": manifest["preprocessing"], "svm_parameters": parameters,
        "train_count": len(y_train), "test_count": len(y_test),
        "feature_dimension": X_train.shape[1],
        "train_correct": int(np.sum(y_train == train_prediction)),
        "test_correct": int(np.sum(y_test == test_prediction)),
        "train_accuracy": float(train_accuracy), "test_accuracy": float(test_accuracy),
        "classification_report": report, "confusion_matrix": matrix.tolist(),
        "class_names": class_names, "converged": converged, "n_iter": int(clf.n_iter_),
        "fit_seconds": fit_seconds, "total_seconds": time.perf_counter() - started,
        "dataset_sha256": manifest["dataset_sha256"],
        "manifest_sha256": sha256(manifest_path), "model_sha256": sha256(model_path),
        "predictions_sha256": sha256(prediction_path), "saved_model_replayed": True,
        "classifier_sha256": sha256(__file__), "sklearn_version": sklearn.__version__,
        "warnings": [str(w.message) for w in captured],
    }
    write_json(output_dir / "results.json", result)
    summary = (
        f"参数组：{args.params}\n输入：224×224原图，不缩放\n"
        f"HOG：{manifest['hog']}\nSVM C：{config['C']}\n"
        f"训练图像：{len(y_train)}；测试图像：{len(y_test)}\n"
        f"训练准确率：{train_accuracy:.4%} ({result['train_correct']}/{len(y_train)})\n"
        f"测试准确率：{test_accuracy:.4%} ({result['test_correct']}/{len(y_test)})\n"
        f"测试宏平均F1：{report['macro avg']['f1-score']:.6f}\n"
        f"训练耗时：{fit_seconds:.3f}s；迭代次数：{clf.n_iter_}；已收敛\n"
        "保存模型的训练和测试预测重放一致。\n"
    )
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(classification_report(y_test, test_prediction, labels=labels,
                                target_names=class_names, digits=4, zero_division=0))
    print(f"Results saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
