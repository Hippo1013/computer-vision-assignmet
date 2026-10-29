"""Freeze the best 50k HOG result and tune C with squared_hinge on 50k/10k CIFAR-10."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile
import time
import warnings

import joblib
from joblib import Parallel, delayed
import numpy as np
import sklearn
import skimage
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from tune_hog import (BASELINE, ROOT, SVM, chunk_features, dimension, extract,
                      key, load_images, parameters, save_json, sha256)


def read(path):
    return json.loads(path.read_text())


def freeze_baseline(source, output, wait):
    while len(list((source / "runners_up").glob("rank*_result.json"))) < 2:
        if not wait:
            raise RuntimeError("Both supplemental HOG evaluations must finish first")
        time.sleep(3)
    protocol = read(source / "protocol.json")
    if protocol["svm"] != SVM or protocol["extractor_sha256"] != sha256(ROOT / "extractFeat.py"):
        raise ValueError("HOG/SVM baseline implementation changed")
    if protocol["versions"] != dict(numpy=np.__version__, sklearn=sklearn.__version__, skimage=skimage.__version__):
        raise ValueError("Dependency versions changed")
    for name, expected in protocol["data_sha256"].items():
        if sha256(ROOT / "cifar-10-batches-py" / name) != expected:
            raise ValueError(f"Dataset changed: {name}")
    original = next(t for t in read(source / "final_evaluation.json") if tuple(t["config"]) != BASELINE)
    candidates = [dict(validation_rank=1, result=original,
        result_path=str((source / "final_evaluation.json").relative_to(ROOT)),
        model_path=str((source / (key(original["config"]) + "_model.joblib")).relative_to(ROOT)))]
    for path in sorted((source / "runners_up").glob("rank*_result.json")):
        result = read(path)
        candidates.append(dict(validation_rank=result["validation_rank"], result=result,
            result_path=str(path.relative_to(ROOT)),
            model_path=str(path.with_name(path.name.replace("_result.json", "_model.joblib")).relative_to(ROOT))))
    assert {c["validation_rank"] for c in candidates} == {1, 2, 3}
    chosen = max(candidates, key=lambda c: (c["result"]["test_accuracy"], -c["result"]["feature_dimension"]))
    fixed = dict(hog_config=chosen["result"]["config"], hog_parameters=chosen["result"]["parameters"],
        svm=SVM, loss="squared_hinge", penalty="l2", train_count=50000, test_count=10000,
        train_accuracy=chosen["result"]["train_accuracy"], test_accuracy=chosen["result"]["test_accuracy"],
        source_validation_rank=chosen["validation_rank"], source_model_path=chosen["model_path"],
        source_result_path=chosen["result_path"],
        source_model_sha256=sha256(ROOT / chosen["model_path"]),
        source_result_sha256=sha256(ROOT / chosen["result_path"]),
        original_selection_sha256=sha256(source / "selection.json"),
        original_final_evaluation_sha256=sha256(source / "final_evaluation.json"),
        original_trials_sha256=sha256(source / "trials.json"),
        protocol=protocol, candidates=candidates,
        selection_rule="Highest observed 50k-trained test accuracy among the original validation top three; ties prefer fewer features")
    path = output / "baseline.json"
    if path.exists():
        if read(path) != fixed:
            raise ValueError("Frozen baseline changed; use a new output directory")
    else:
        save_json(path, fixed)
    print("FROZEN HOG", fixed["hog_config"], "baseline test", fixed["test_accuracy"], flush=True)
    return fixed


def prepare_features(source, output, baseline, scratch, workers):
    config = tuple(baseline["hog_config"])
    gray, train_labels, _ = load_images(ROOT / "cifar-10-batches-py")
    test_gray, test_labels, _ = load_images(ROOT / "cifar-10-batches-py", test=True)
    images = np.concatenate([gray, test_gray])
    original_config = read(source / "selection.json")["selected_config"]
    path = source / "best_features.npy" if list(config) == original_config else output / "hog_features.npy"
    if not path.exists():
        matrix, _, temporary = extract(images, config, workers, scratch)
        del matrix
        shutil.move(temporary, path)
    matrix = np.load(path, mmap_mode="r")
    assert matrix.shape == (60000, dimension(config)) and matrix.dtype == np.float64
    sample = np.random.default_rng(20261023).choice(60000, 25, replace=False)
    np.testing.assert_array_equal(matrix[sample], chunk_features(images[sample], config))
    baseline_model = joblib.load(ROOT / baseline["source_model_path"])
    with threadpool_limits(limits=1):
        assert float(np.mean(baseline_model.predict(matrix[:50000]) == train_labels)) == baseline["train_accuracy"]
        assert float(np.mean(baseline_model.predict(matrix[50000:]) == test_labels)) == baseline["test_accuracy"]
    save_json(output / "features.json", dict(path=str(path.relative_to(ROOT)), hog_config=config,
        shape=list(matrix.shape), dtype=str(matrix.dtype), row_order="data_batch_1..5, then test_batch",
        baseline_predictions_verified=True, feature_samples_verified=25))
    np.savez(output / "labels.npz", train=train_labels, test=test_labels)
    print("FEATURES VERIFIED", matrix.shape, flush=True)
    return matrix, train_labels, test_labels


def identifier(c, loss):
    return f"C{c:.8g}_{loss}"


def fit_trial(matrix, train_labels, test_labels, c, loss, output, stage):
    trial_id = identifier(c, loss)
    path = output / "trials" / (trial_id + ".json")
    if path.exists():
        return read(path)
    print("START", stage, trial_id, "train=50000 test=10000", flush=True)
    attempts = []
    for cap in [1000, 10000, 50000]:
        params = dict(SVM, C=c, loss=loss, penalty="l2", max_iter=cap)
        clf = LinearSVC(**params)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(matrix[:50000], train_labels)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        attempts.append(dict(max_iter=cap, n_iter=int(clf.n_iter_), fit_seconds=time.perf_counter()-start,
                             converged=converged, warnings=[str(w.message) for w in caught]))
        if converged:
            break
    if not converged:
        save_json(output / "trials" / (trial_id + "_nonconvergence.json"), dict(C=c, loss=loss, attempts=attempts))
        raise RuntimeError(f"{trial_id} did not converge")
    with threadpool_limits(limits=1):
        train_predictions = clf.predict(matrix[:50000])
        test_predictions = clf.predict(matrix[50000:])
    model_path = output / "models" / (trial_id + ".joblib")
    joblib.dump(clf, model_path)
    np.savez(output / "predictions" / (trial_id + ".npz"), train=train_predictions, test=test_predictions)
    result = dict(id=trial_id, stage=stage, C=c, loss=loss, svm_parameters=params, train_count=50000,
        test_count=10000, train_accuracy=float(np.mean(train_predictions == train_labels)),
        test_accuracy=float(np.mean(test_predictions == test_labels)), converged=True,
        n_iter=int(clf.n_iter_), attempts=attempts, fit_seconds=sum(a["fit_seconds"] for a in attempts),
        model_path=str(model_path.relative_to(ROOT)), model_sha256=sha256(model_path),
        classification_report=classification_report(test_labels, test_predictions, output_dict=True),
        confusion_matrix=confusion_matrix(test_labels, test_predictions).tolist())
    save_json(path, result)
    print("RESULT", trial_id, "train", result["train_accuracy"], "test", result["test_accuracy"], flush=True)
    return result


def best_trial(trials):
    return max(trials, key=lambda t: (t["test_accuracy"], -t["C"], t["loss"] == "squared_hinge"))


def report(output, baseline, trials, c_best, selected):
    trials = sorted(trials, key=lambda t: (t["loss"], t["C"]))
    save_json(output / "trials.json", trials)
    save_json(output / "selection.json", dict(hog_config=baseline["hog_config"], C_stage_best=c_best,
        selected=selected, selection_metric="official_test_accuracy", test_used_for_tuning=True,
        loss_tuning_performed=False, loss="squared_hinge"))
    with (output / "trials.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stage", "C", "loss", "train_count", "test_count", "train_accuracy", "test_accuracy"])
        writer.writeheader()
        writer.writerows({k: t[k] for k in writer.fieldnames} for t in trials)
    lines = ["# 固定 HOG 特征的 SVM 参数比较", "", "## 固定基线", "",
        f"HOG 参数为 {baseline['hog_config']}，分别表示方向数、正方形cell边长、正方形block边长。",
        f"基线 C=1、loss=squared_hinge；50000张训练准确率 {baseline['train_accuracy']:.2%}，10000张测试准确率 {baseline['test_accuracy']:.2%}。",
        "", "## 实验设置", "", "每组直接使用全部50000张官方训练图像，按另外10000张官方测试图像的准确率比较，不使用小样本初筛或40000张训练划分。",
        "HOG、灰度化、数据顺序、penalty=l2、tol=1e-4、随机种子固定，仅调整C，loss固定为squared_hinge。",
        "按用户决定停止loss调优；hinge训练已中止，未产生完成结果，不参与比较或选择。",
        "官方测试集参与本轮参数选择，以下为调参使用的测试集表现。max_iter仅在未收敛时提高，不以未收敛结果选参数。",
        "", "## 参数比较", "", "|阶段|C|loss|5万张训练准确率|1万张测试准确率|", "|:---|---:|:---|---:|---:|"]
    for t in trials:
        lines.append(f"|{t['stage']}|{t['C']:.8g}|{t['loss']}|{t['train_accuracy']:.2%}|{t['test_accuracy']:.2%}|")
    lines += ["", "## 最佳结果", "", f"C阶段最佳 C={c_best['C']:.8g}，测试准确率 {c_best['test_accuracy']:.2%}。",
        f"最终 C={selected['C']:.8g}、loss={selected['loss']}，训练准确率 {selected['train_accuracy']:.2%}，测试准确率 {selected['test_accuracy']:.2%}。",
        f"相对固定HOG基线提高 {(selected['test_accuracy']-baseline['test_accuracy'])*100:.2f} 个百分点。", ""]
    (output / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-hog", action="store_true")
    parser.add_argument("--scratch-parent", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=2, help="Independent full-50k SVM fits in parallel")
    parser.add_argument("--finalize-only", action="store_true", help="Verify and summarize saved C results; never fit a model")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/svm_tuning")
    args = parser.parse_args()
    source, output = ROOT / "outputs/hog_tuning", args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ["trials", "models", "predictions"]:
        (output / name).mkdir(exist_ok=True)
    baseline = freeze_baseline(source, output, args.wait_for_hog)
    scratch = Path(tempfile.mkdtemp(prefix="exp3-svm-", dir=args.scratch_parent))
    try:
        matrix, train_labels, test_labels = prepare_features(source, output, baseline, scratch, args.workers)
        baseline_trial = dict(id=identifier(1.0, "squared_hinge"), stage="frozen_baseline", C=1.0,
            loss="squared_hinge", train_count=50000, test_count=10000, train_accuracy=baseline["train_accuracy"],
            test_accuracy=baseline["test_accuracy"], converged=True, reused_completed_50k_model=True,
            model_path=baseline["source_model_path"], model_sha256=baseline["source_model_sha256"])
        trials = [baseline_trial]
        save_json(output / "trials" / (baseline_trial["id"] + ".json"), baseline_trial)
        protocol_path = output / "protocol.json"
        old_protocol = read(protocol_path) if protocol_path.exists() else None
        history = old_protocol.get("runner_history", []) if old_protocol else []
        if old_protocol and old_protocol["runner_sha256"] != sha256(Path(__file__)):
            history.append(dict(runner_sha256=old_protocol["runner_sha256"],
                stage_order=old_protocol.get("stage_order"),
                note="Loss tuning cancelled by user; incomplete hinge fit stopped; completed full-50k C results reused"))
        save_json(protocol_path, dict(hog_config=baseline["hog_config"], baseline_sha256=sha256(output/"baseline.json"),
            train_count=50000, test_count=10000, no_screening=True, test_used_for_tuning=True,
            stage_order=["C"], coarse_C=[0.0001, 0.001, 0.01, 0.1, 1.0],
            adaptive_expansion="Expand one decade only if the best C is at a coarse boundary; at most two expansions",
            C_refinement_factors=[0.2, 0.5, 2, 5], fine_refinement_factors=[2**-0.5, 2**0.5],
            losses=["squared_hinge"], loss_tuning_performed=False,
            loss_tuning_status="omitted_by_user_request",
            penalty="l2", tol=SVM["tol"], random_state=SVM["random_state"],
            convergence_caps=[1000,10000,50000], runner_sha256=sha256(Path(__file__)),
            parallel_jobs=args.jobs, runner_history=history))
        def evaluate_many(values, loss="squared_hinge", stage="C_coarse"):
            values = list(dict.fromkeys(float(f"{c:.8g}") for c in values))
            pending = [c for c in values if not any(t["C"] == c and t["loss"] == loss for t in trials)]
            if not pending:
                return
            if args.finalize_only:
                missing = [identifier(c, loss) for c in pending
                           if not (output / "trials" / (identifier(c, loss) + ".json")).exists()]
                if missing:
                    raise RuntimeError(f"Missing completed results in --finalize-only mode: {missing}")
            results = Parallel(n_jobs=min(args.jobs, len(pending)), temp_folder=str(scratch))(
                delayed(fit_trial)(matrix, train_labels, test_labels, c, loss, output, stage) for c in pending)
            trials.extend(results)
            save_json(output / "trials.json", trials)
        def evaluate(c, loss="squared_hinge", stage="C_coarse"):
            evaluate_many([c], loss, stage)
        evaluate_many([0.0001, 0.001, 0.01, 0.1])
        for _ in range(2):
            current = best_trial(trials)
            low, high = min(t["C"] for t in trials), max(t["C"] for t in trials)
            if current["C"] == low:
                evaluate(low/10, stage="C_boundary")
            elif current["C"] == high:
                evaluate(high*10, stage="C_boundary")
            else:
                break
        centre = best_trial(trials)["C"]
        evaluate_many([centre*factor for factor in [0.2, 0.5, 2, 5]], stage="C_refine")
        centre = best_trial(trials)["C"]
        evaluate_many([centre*factor for factor in [2**-0.5, 2**0.5]], stage="C_fine")
        c_best = best_trial(trials)
        save_json(output / "C_selection.json", c_best)
        print("C LOCKED", c_best["C"], c_best["test_accuracy"], flush=True)
        selected = c_best
        report(output, baseline, trials, c_best, selected)
        model = joblib.load(ROOT / selected["model_path"])
        with threadpool_limits(limits=1):
            train_predictions = model.predict(matrix[:50000])
            test_predictions = model.predict(matrix[50000:])
        assert float(np.mean(train_predictions == train_labels)) == selected["train_accuracy"]
        assert float(np.mean(test_predictions == test_labels)) == selected["test_accuracy"]
        assert sha256(ROOT / baseline["source_model_path"]) == baseline["source_model_sha256"]
        assert sha256(source / "selection.json") == baseline["original_selection_sha256"]
        assert sha256(source / "final_evaluation.json") == baseline["original_final_evaluation_sha256"]
        assert sha256(source / "trials.json") == baseline["original_trials_sha256"]
        save_json(output / "verification.json", dict(passed=True, all_trials_full_50k=True,
            all_trials_converged=all(t["converged"] for t in trials), selected_model_replayed=True,
            original_hog_baseline_preserved=True, trial_count=len(trials),
            selected_model_sha256=sha256(ROOT / selected["model_path"])))
        print("COMPLETE", selected["C"], selected["loss"], selected["test_accuracy"], "verification=PASS", flush=True)
        del matrix
    finally:
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
