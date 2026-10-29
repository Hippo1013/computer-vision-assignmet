"""CIFAR-10线性SVM训练与测试，--params old/new选择原始或调优参数。"""

import argparse
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

from extractFeat import CONFIGS, feature_dimension, feature_directory


def load_features(directory, dimension):
    paths = sorted(directory.glob('*.feat'))
    if not paths:
        raise FileNotFoundError(f'没有特征文件，请先运行对应参数组的 extractFeat.py：{directory}')
    features = np.empty((len(paths), dimension), dtype=np.float64)
    labels = np.empty(len(paths), dtype=np.int64)
    for index, path in enumerate(paths):
        data = joblib.load(path)
        if (data.shape != (dimension + 1,) or not np.isfinite(data).all()
                or data[-1] != int(data[-1]) or not 0 <= int(data[-1]) <= 9):
            raise ValueError(f'特征维度、数值或标签与当前参数不匹配：{path}')
        features[index] = data[:-1]
        labels[index] = int(data[-1])
    return features, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', choices=CONFIGS, default='old', help='参数组，默认old')
    parser.add_argument('--feature-root', type=Path,
                        help='特征根目录，包含train/test；默认data/features/{old,new}')
    args = parser.parse_args()
    root = feature_directory(args.params, args.feature_root)
    dimension = feature_dimension(args.params)
    manifest_path = root / 'manifest.json'
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['parameter_set'] != args.params or manifest['feature_dimension'] != dimension:
            raise ValueError('特征清单与 --params 不匹配，请选择对应参数组')
    started = time.time()
    print(f'参数组：{args.params}；SVM C={CONFIGS[args.params]["C"]}；特征目录：{root}',
          flush=True)
    train, train_labels = load_features(root / 'train', dimension)
    test, test_labels = load_features(root / 'test', dimension)
    if manifest is not None and (len(train) != manifest['train_count']
                                 or len(test) != manifest['test_count']):
        raise ValueError('特征数量与提取清单不一致，请重新完整提取')
    clf = LinearSVC(max_iter=1000, C=CONFIGS[args.params]['C'], random_state=20261022)
    print(f'Training a Linear SVM Classifier: {len(train)} train / {len(test)} test...',
          flush=True)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', ConvergenceWarning)
        clf.fit(train, train_labels)
    for warning in captured:
        print(f'Warning: {warning.message}', flush=True)
    train_accuracy = float(np.mean(clf.predict(train) == train_labels))
    correct = int(np.sum(clf.predict(test) == test_labels))
    test_accuracy = correct / len(test_labels)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in captured)
    print(f'训练准确率：{train_accuracy:.4%}', flush=True)
    print(f'测试准确率：{test_accuracy:.4%} ({correct}/{len(test_labels)})', flush=True)
    print(f'收敛状态：{"已收敛" if converged else "未收敛"}；迭代次数：{clf.n_iter_}', flush=True)
    print(f'total time is: {time.time() - started:.3f}s', flush=True)


if __name__ == '__main__':
    main()
