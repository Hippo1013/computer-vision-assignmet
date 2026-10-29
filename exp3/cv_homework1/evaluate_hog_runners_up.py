"""Retrain validation ranks 2/3 on all 50k images, preserving the original selection."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import joblib
import numpy as np
import sklearn
import skimage
from sklearn.metrics import classification_report, confusion_matrix

from tune_hog import (ROOT, SVM, chunk_features, dimension, extract, fit_score,
                      key, load_images, parameters, save_json, sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=[2, 3], required=True)
    parser.add_argument("--source-output", type=Path, default=ROOT / "outputs/hog_tuning")
    parser.add_argument("--scratch-parent", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    source = args.source_output.resolve()
    protocol = json.loads((source / "protocol.json").read_text())
    trials = sorted([t for t in json.loads((source / "trials.json").read_text())
                     if t["phase"] == "full" and t["converged"]],
                    key=lambda t: (-t["validation_accuracy"], t["feature_dimension"]))
    selection = json.loads((source / "selection.json").read_text())
    if trials[0]["config"] != selection["selected_config"]:
        raise ValueError("Original validation selection no longer matches the trial ranking")
    candidate = trials[args.rank - 1]
    config = tuple(candidate["config"])
    data_dir = ROOT / "cifar-10-batches-py"
    versions = dict(numpy=np.__version__, sklearn=sklearn.__version__, skimage=skimage.__version__)
    if SVM != protocol["svm"] or versions != protocol["versions"]:
        raise ValueError("SVM settings or dependencies changed")
    if sha256(ROOT / "extractFeat.py") != protocol["extractor_sha256"]:
        raise ValueError("Original feature extractor changed")
    for name, expected in protocol["data_sha256"].items():
        if sha256(data_dir / name) != expected:
            raise ValueError(f"Dataset changed: {name}")
    frozen_paths = [source / name for name in ["protocol.json", "trials.json", "selection.json", "final_evaluation.json"]]
    frozen_paths += list(source.glob("*_model.joblib")) + list(source.glob("*_test_predictions.npy"))
    frozen = {str(path.relative_to(source)): sha256(path) for path in frozen_paths}
    output = source / "runners_up"
    output.mkdir(exist_ok=True)
    stem = f"rank{args.rank}_{key(config)}"
    result_path = output / (stem + "_result.json")
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous["original_artifact_sha256"] != frozen or previous["config"] != list(config):
            raise ValueError("Existing supplemental result has incompatible provenance")
        print("ALREADY COMPLETE", stem, previous["test_accuracy"], flush=True)
        return
    save_json(output / (stem + "_protocol.json"), dict(
        validation_rank=args.rank, config=config, validation_trial=candidate,
        train_count=50000, test_count=10000, svm=SVM, versions=versions,
        extractor_sha256=protocol["extractor_sha256"], data_sha256=protocol["data_sha256"],
        runner_sha256=sha256(Path(__file__)), python=sys.executable,
        role="User-requested supplemental evaluation of validation ranks 2 and 3",
        original_artifact_sha256=frozen, original_selection_preserved=True))
    scratch = Path(tempfile.mkdtemp(prefix="exp3-hog-rank-", dir=args.scratch_parent))
    try:
        print("START", stem, "train=50000 test=10000", flush=True)
        gray, labels, _ = load_images(data_dir)
        test_gray, test_labels, _ = load_images(data_dir, test=True)
        images = np.concatenate([gray, test_gray])
        all_labels = np.concatenate([labels, test_labels])
        matrix, extraction_seconds, _ = extract(images, config, args.workers, scratch)
        clf, metrics = fit_score(matrix, all_labels, np.arange(50000), np.arange(50000, 60000))
        metrics["test_accuracy"] = metrics.pop("validation_accuracy")
        if not metrics["converged"]:
            raise RuntimeError("SVM did not converge")
        train_predictions = clf.predict(matrix[:50000])
        test_predictions = clf.predict(matrix[50000:])
        np.savez(output / (stem + "_predictions.npz"), train=train_predictions, test=test_predictions)
        model_path = output / (stem + "_model.joblib")
        joblib.dump(clf, model_path)
        # Verify serialized model, saved predictions, metrics and feature identity before removing caches.
        loaded = joblib.load(model_path)
        for field, value in SVM.items():
            assert loaded.get_params()[field] == value
        np.testing.assert_array_equal(loaded.predict(matrix[50000:]), test_predictions)
        saved = np.load(output / (stem + "_predictions.npz"))
        np.testing.assert_array_equal(saved["train"], train_predictions)
        np.testing.assert_array_equal(saved["test"], test_predictions)
        assert float(np.mean(saved["train"] == labels)) == metrics["train_accuracy"]
        assert float(np.mean(saved["test"] == test_labels)) == metrics["test_accuracy"]
        sample = np.random.default_rng(20261023).choice(60000, 25, replace=False)
        np.testing.assert_array_equal(matrix[sample], chunk_features(images[sample], config))
        confusion = confusion_matrix(test_labels, test_predictions)
        assert confusion.trace() == np.count_nonzero(test_labels == test_predictions)
        for path in frozen_paths:
            assert sha256(path) == frozen[str(path.relative_to(source))]
        result = dict(validation_rank=args.rank, config=config, parameters=parameters(config),
            validation_accuracy_40k=candidate["validation_accuracy"], feature_dimension=dimension(config),
            train_count=50000, test_count=10000, extraction_seconds=extraction_seconds, **metrics,
            classification_report=classification_report(test_labels, test_predictions, output_dict=True),
            confusion_matrix=confusion.tolist(), original_artifact_sha256=frozen,
            original_selection_preserved=True, verification_passed=True)
        save_json(result_path, result)
        print("RESULT", stem, metrics, "verification=PASS", flush=True)
        del matrix
    finally:
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
