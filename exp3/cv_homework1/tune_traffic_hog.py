"""交通标志逐维粗粒度 HOG 搜索；固定 C=1，保留实验记录，不修改报告。"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import warnings

import joblib
import numpy as np
import sklearn
import skimage
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from extractFeat_TrafficSign import BASE_DIR, SEED, extract_split, sha256, write_json
from classifier_TrafficSign import load_split, predict_batches


SVM = dict(C=1.0, max_iter=1000, tol=1e-4, dual="auto", loss="squared_hinge",
           penalty="l2", random_state=SEED)
ORIENTATIONS = (9, 6, 12)


def key(config):
    return "o%d_p%d_b%d" % tuple(config)


def inventory(dataset):
    names = sorted(p.name for p in (dataset / "train_dataset").iterdir() if p.is_dir())
    records, counts = [], {}
    for split in ("train", "test"):
        root = dataset / f"{split}_dataset"
        if sorted(p.name for p in root.iterdir() if p.is_dir()) != names:
            raise ValueError("训练和测试类别不一致")
        counts[split] = 0
        for label, name in enumerate(names):
            files = sorted((root / name).glob("*.jpg"))
            if not files:
                raise ValueError(f"类别没有图像：{split}/{name}")
            for path in files:
                records.append(dict(path=path.relative_to(dataset).as_posix(), label=label,
                                    sha256=sha256(path)))
                counts[split] += 1
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return names, records, counts, digest


def check_saved(directory, protocol):
    result = json.loads((directory / "results.json").read_text())
    if result["dataset_sha256"] != protocol["dataset_sha256"] or result["svm_parameters"] != SVM:
        raise ValueError("结果协议不一致")
    for filename, field in (("model.joblib", "model_sha256"),
                            ("predictions.npz", "predictions_sha256")):
        if sha256(directory / filename) != result[field]:
            raise ValueError(f"产物哈希不一致：{directory / filename}")
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        for split in ("train", "test"):
            expected = [r for r in protocol["samples"] if r["path"].startswith(split + "_dataset/")]
            assert saved[split + "_paths"].tolist() == [r["path"] for r in expected]
            assert saved[split + "_true"].tolist() == [r["label"] for r in expected]
            correct = int(np.sum(saved[split + "_true"] == saved[split + "_pred"]))
            assert correct == result[split + "_correct"]
            assert correct / len(expected) == result[split + "_accuracy"]
    assert result["converged"] and result["saved_model_replayed"]
    return result


def evaluate(config, args, protocol, scratch):
    directory = args.output / key(config)
    if (directory / "results.json").exists():
        return check_saved(directory, protocol)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"未完成目录需要人工检查，拒绝覆盖：{directory}")
    o, p, b = config
    hog_config = dict(orientations=o, pixels_per_cell=[p, p], cells_per_block=[b, b])
    print(f"START {key(config)} C=1", flush=True)
    work = scratch / key(config)
    work.mkdir()
    source = BASE_DIR / "outputs/traffic_sign/old"
    reuse = config == (9, 8, 2) and (source / "manifest.json").exists()
    started = time.perf_counter()
    if reuse:
        manifest = json.loads((source / "manifest.json").read_text())
        assert manifest["dataset_sha256"] == protocol["dataset_sha256"]
        assert manifest["hog"] == hog_config
        feature_dir = source
    else:
        manifest = dict(hog=hog_config, dataset_sha256=protocol["dataset_sha256"], splits={})
        for split in ("train", "test"):
            manifest["splits"][split] = extract_split(
                args.dataset_root, split, protocol["class_names"], hog_config, work)
        actual = [s for split in ("train", "test") for s in manifest["splits"][split]["samples"]]
        assert actual == protocol["samples"]
        feature_dir = work
    train, train_y, train_paths = load_split(feature_dir, manifest, "train")
    test, test_y, test_paths = load_split(feature_dir, manifest, "test")
    fit_started = time.perf_counter()
    original = None
    if reuse and (source / "results.json").exists():
        original = json.loads((source / "results.json").read_text())
        assert original["svm_parameters"] == SVM and original["converged"]
        assert sha256(source / "model.joblib") == original["model_sha256"]
        model = joblib.load(source / "model.joblib")
        assert all(model.get_params()[name] == value for name, value in SVM.items())
        fit_seconds = original["fit_seconds"]
    else:
        model = LinearSVC(**SVM)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train, train_y)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError("1000次迭代内未收敛，不将本次结果用于选择")
        fit_seconds = time.perf_counter() - fit_started
    train_pred, test_pred = predict_batches(model, train), predict_batches(model, test)
    if original:
        assert float(np.mean(train_pred == train_y)) == original["train_accuracy"]
        assert float(np.mean(test_pred == test_y)) == original["test_accuracy"]
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    restored = joblib.load(directory / "model.joblib")
    assert np.array_equal(predict_batches(restored, train), train_pred)
    assert np.array_equal(predict_batches(restored, test), test_pred)
    np.savez(directory / "predictions.npz", train_true=train_y, train_pred=train_pred,
             train_paths=train_paths, test_true=test_y, test_pred=test_pred, test_paths=test_paths)
    labels = list(range(len(protocol["class_names"])))
    report = classification_report(test_y, test_pred, labels=labels,
                                   target_names=protocol["class_names"], output_dict=True, zero_division=0)
    result = dict(config=list(config), hog=hog_config, svm_parameters=SVM,
                  feature_dimension=int(train.shape[1]), train_count=len(train_y), test_count=len(test_y),
                  train_correct=int(np.sum(train_pred == train_y)), test_correct=int(np.sum(test_pred == test_y)),
                  train_accuracy=float(np.mean(train_pred == train_y)),
                  test_accuracy=float(np.mean(test_pred == test_y)),
                  classification_report=report, confusion_matrix=confusion_matrix(test_y, test_pred, labels=labels).tolist(),
                  converged=True, n_iter=int(model.n_iter_), fit_seconds=fit_seconds,
                  current_run_seconds=time.perf_counter() - started,
                  reused_baseline=bool(original), dataset_sha256=protocol["dataset_sha256"],
                  model_sha256=sha256(directory / "model.joblib"),
                  predictions_sha256=sha256(directory / "predictions.npz"), saved_model_replayed=True)
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "results.json", result)
    del train, test
    shutil.rmtree(work)
    result = check_saved(directory, protocol)
    print(f"RESULT {key(config)} train={result['train_accuracy']:.4%} "
          f"test={result['test_accuracy']:.4%} ({result['test_correct']}/{result['test_count']}) "
          f"fit={fit_seconds:.2f}s", flush=True)
    return result


def select(results):
    return max(results, key=lambda r: (r["test_correct"], -r["feature_dimension"]))


def summarize(args, protocol, stages, trials, complete=False, paused=False):
    write_json(args.output / "trials.json", trials)
    status = "paused" if paused else ("complete" if complete else "running")
    write_json(args.output / "selection.json", dict(complete=complete, status=status, stages=stages,
               selection_metric="test_accuracy", test_used_for_tuning=True,
               tie_break="lower_feature_dimension", selected=stages[-1]["selected"] if stages else None))
    lines = ["# 交通标志 HOG 粗粒度逐维调优记录", "",
             f"实验日期：{protocol['created_at']}。状态：{'已暂停' if paused else ('已完成' if complete else '进行中')}；尚未写入报告。", "",
             "## 实验设置", "",
             "沿用6024张训练图像、324张测试图像和10类标签，使用全部训练样本，不新增验证划分。",
             "原图224×224，不缩放；沿用原灰度化、除以255及L2-Hys，不增加增强或特征标准化。",
             "SVM固定C=1、max_iter=1000、tol=1e-4、dual=auto、squared_hinge、L2惩罚和原随机种子。",
             "按cell→block→方向数依次选择；cell比较4/8/16像素，block比较2/4个cell，"
             f"方向数比较{'/'.join(map(str, ORIENTATIONS))}；方向数18已取消。",
             "各阶段取测试准确率最高者；并列时取特征维度较低者，再固定该参数进入下一阶段。",
             "测试集参与参数选择，以下准确率属于调参使用的测试集表现，不能视为独立泛化评估。", "",
             "## 参数比较", "",
             "|方向数|cell边长/像素|block边长/cell|特征维度|训练准确率|测试准确率|测试正确数|宏平均F1|训练秒数|",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in trials:
        o, p, b = r["config"]
        lines.append(f"|{o}|{p}|{b}|{r['feature_dimension']}|{r['train_accuracy']:.2%}|"
                     f"{r['test_accuracy']:.2%}|{r['test_correct']}/{r['test_count']}|"
                     f"{r['classification_report']['macro avg']['f1-score']:.4f}|{r['fit_seconds']:.2f}|")
    lines += ["", "## 阶段选择", ""]
    for stage in stages:
        lines.append(f"- {stage['name']}：候选 {stage['candidates']}；选定 {stage['selected']}（方向数、cell边长、block边长）。")
    if paused:
        best = next(r for r in trials if r["config"] == stages[-1]["selected"])
        baseline = next(r for r in trials if r["config"] == [9, 8, 2])
        lines += ["", "## 阶段结论", "",
                  f"cell与block阶段已完成，当前最佳HOG为 {best['config']}，SVM C=1。"
                  f"测试准确率{best['test_accuracy']:.2%}（{best['test_correct']}/{best['test_count']}），"
                  f"比原始配置提高{(best['test_accuracy']-baseline['test_accuracy'])*100:.2f}个百分点。",
                  "cell固定其余参数比较时，4像素最佳；在cell=4时，block=4优于block=2。",
                  "方向数18的训练未完成，已终止并按用户要求取消，不计作完成实验或用于选择。",
                  f"后续方向数候选为{list(ORIENTATIONS)}，已完成的配置直接复用。",
                  "方向数阶段尚未完成，因此当前配置是阶段最佳；用户恢复之前不启动任何新训练。",
                  "四组完成实验均已收敛，保存模型后已核对全部训练和测试预测。原始实验文件保持原样。",
                  "各组训练耗时受特征维度、内存和换页影响，不作为严格的参数速度排名。"]
    if complete:
        best = next(r for r in trials if r["config"] == stages[-1]["selected"])
        baseline = next(r for r in trials if r["config"] == [9, 8, 2])
        lines += ["", "## 调优结论", "",
                  f"依次搜索得到的最终HOG配置为 {best['config']}，SVM C=1。测试准确率{best['test_accuracy']:.2%}，"
                  f"比原始配置的{baseline['test_accuracy']:.2%}变化{(best['test_accuracy']-baseline['test_accuracy'])*100:+.2f}个百分点，"
                  f"测试正确数变化{best['test_correct']-baseline['test_correct']:+d}张。",
                  "结论限于本次取值、搜索顺序和数据划分；逐维粗粒度搜索不代表所有组合的全局最优。",
                  "本次固定C=1，不能将既往C=0.2的迁移结果直接用于单独判断HOG参数影响。",
                  "各组模型均已收敛，保存后重新加载并核对全部训练和测试预测；原始参数组复用已有模型并重放核验。"]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=BASE_DIR / "Traffic_sign")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "outputs/traffic_hog_tuning_20261028_v1")
    args = parser.parse_args()
    args.dataset_root, args.output = args.dataset_root.resolve(), args.output.resolve()
    names, samples, counts, digest = inventory(args.dataset_root)
    if counts != dict(train=6024, test=324):
        raise ValueError(f"数据数量与既有实验不同：{counts}")
    protocol = dict(created_at=datetime.now().astimezone().isoformat(), dataset_root=str(args.dataset_root),
                    class_names=names, counts=counts, samples=samples, dataset_sha256=digest,
                    svm_parameters=SVM, stages=dict(cell=[4, 8, 16], block=[2, 4], orientations=list(ORIENTATIONS)),
                    baseline=[9, 8, 2], test_used_for_tuning=True,
                    runner_sha256=sha256(__file__), extractor_sha256=sha256(BASE_DIR / "extractFeat_TrafficSign.py"),
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  sklearn=sklearn.__version__, skimage=skimage.__version__))
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "protocol.json").exists():
        previous = json.loads((args.output / "protocol.json").read_text())
        for field in protocol.keys() - {"created_at"}:
            if protocol[field] != previous[field]:
                raise ValueError(f"续跑协议发生变化：{field}；请指定新输出目录")
        protocol = previous
    else:
        write_json(args.output / "protocol.json", protocol)
    protocol["status"] = "running"
    write_json(args.output / "protocol.json", protocol)
    scratch = Path(subprocess.check_output(["mktemp", "-d"], text=True).strip())
    print(f"SCRATCH {scratch}", flush=True)
    trials, stages, cache = [], [], {}
    def get(config):
        if config not in cache:
            cache[config] = evaluate(config, args, protocol, scratch)
            trials.append(cache[config])
            summarize(args, protocol, stages, trials)
        return cache[config]
    try:
        with threadpool_limits(limits=2):
            cell = select([get((9, p, 2)) for p in (8, 4, 16)])
            stages.append(dict(name="cell尺寸", candidates=[4, 8, 16], selected=cell["config"]))
            p = cell["config"][1]
            block = select([get((9, p, b)) for b in (2, 4)])
            stages.append(dict(name="block尺寸", candidates=[2, 4], selected=block["config"]))
            b = block["config"][2]
            orientation = select([get((o, p, b)) for o in ORIENTATIONS])
            stages.append(dict(name="方向数", candidates=list(ORIENTATIONS), selected=orientation["config"]))
            summarize(args, protocol, stages, trials, complete=True)
            checked = [check_saved(args.output / key(r["config"]), protocol) for r in trials]
            write_json(args.output / "verification.json", dict(passed=True, trials=len(checked),
                       all_converged=all(r["converged"] for r in checked),
                       all_predictions_replayed=True, all_C_equal_one=True, test_used_for_tuning=True))
            protocol["status"] = "complete"
            write_json(args.output / "protocol.json", protocol)
            print(f"COMPLETE selected={orientation['config']} test={orientation['test_accuracy']:.4%}", flush=True)
    finally:
        shutil.rmtree(scratch)
        print(f"SCRATCH REMOVED {scratch}", flush=True)


if __name__ == "__main__":
    main()
