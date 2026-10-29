from skimage.feature import hog
import argparse
import json
from pathlib import Path
import numpy as np
import joblib
import os
import time
import pickle


def mkdir_or_exist(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# 数据读取
def getData(filePath):
    TrainData = []
    for childDir in os.listdir(filePath):
        if 'data_batch' in childDir:
            f = os.path.join(filePath, childDir)
            data = unpickle(f)
            train = np.reshape(data[b'data'], (10000, 3, 32 * 32))
            labels = np.reshape(data[b'labels'], (10000, 1))
            fileNames = np.reshape(data[b'filenames'], (10000, 1))
            datalebels = zip(train, labels, fileNames)
            TrainData.extend(datalebels)
        if childDir == "test_batch":
            f = os.path.join(filePath, childDir)
            data = unpickle(f)
            test = np.reshape(data[b'data'], (10000, 3, 32 * 32))
            labels = np.reshape(data[b'labels'], (10000, 1))
            fileNames = np.reshape(data[b'filenames'], (10000, 1))
            TestData = zip(test, labels, fileNames)
    return TrainData, TestData


BASE_DIR = Path(__file__).resolve().parent
CONFIGS = {
    'old': dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), C=1.0),
    'new': dict(orientations=9, pixels_per_cell=(4, 4), cells_per_block=(4, 4), C=0.2),
}


def feature_dimension(params):
    config = CONFIGS[params]
    p, b = config['pixels_per_cell'][0], config['cells_per_block'][0]
    return (32 // p - b + 1) ** 2 * b ** 2 * config['orientations']


def feature_directory(params, feature_root=None):
    return (Path(feature_root) if feature_root is not None
            else BASE_DIR / 'data' / 'features' / params).resolve()


# 特征提取
def getFeat(TrainData, TestData, params='old', feature_root=None):
    config = CONFIGS[params]
    orientations = config['orientations']
    pixels_per_cell = config['pixels_per_cell']
    cells_per_block = config['cells_per_block']
    root = feature_directory(params, feature_root)
    train_path, test_path = root / 'train', root / 'test'
    for directory in (train_path, test_path):
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob('*.feat'):
            if joblib.load(path).shape != (feature_dimension(params) + 1,):
                raise ValueError(f'已有特征与 {params} 参数不匹配，请使用其他 --feature-root：{path}')
            break
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        manifest_path.unlink()
    train_count = test_count = 0
    print(f'参数组：{params}；HOG：{orientations}/{pixels_per_cell}/{cells_per_block}；'
          f'特征维度：{feature_dimension(params)}', flush=True)
    # 测试数据
    for data in TestData:
        image = np.reshape(data[0].T, (32, 32, 3))
        gray = rgb2gray(image) / 255.0

        fd = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block)

        # 添加标签和保存图像
        fd = np.concatenate((fd, data[1]))
        filename = list(data[2])
        fd_name = str(filename[0], encoding="utf-8").split('.')[0] + '.feat'
        fd_path = os.path.join(test_path, fd_name)
        joblib.dump(fd, fd_path)
        test_count += 1
    print("Test features are extracted and saved.")
    # 训练数据
    for data in TrainData:
        image = np.reshape(data[0].T, (32, 32, 3))
        gray = rgb2gray(image) / 255.0

        fd = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block)

        # 添加标签和保存图像
        fd = np.concatenate((fd, data[1]))
        filename = list(data[2])
        fd_name = str(filename[0], encoding="utf-8").split('.')[0] + '.feat'
        fd_path = os.path.join(train_path, fd_name)
        joblib.dump(fd, fd_path)
        train_count += 1
    print("Train features are extracted and saved.")
    manifest = dict(parameter_set=params, hog={k: v for k, v in config.items() if k != 'C'},
                    feature_dimension=feature_dimension(params), train_count=train_count,
                    test_count=test_count, image_size=[32, 32],
                    preprocessing='rgb2gray / 255.0; L2-Hys; float64')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                             encoding='utf-8')
    print(f'Train: {train_count}; test: {test_count}; features saved: {root}', flush=True)


# 图像灰度化
def rgb2gray(im):
    gray = im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140
    return gray


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-10 HOG特征提取')
    parser.add_argument('--params', choices=CONFIGS, default='old', help='参数组，默认old')
    parser.add_argument('--data-dir', type=Path, default=BASE_DIR / 'cifar-10-batches-py')
    parser.add_argument('--feature-root', type=Path,
                        help='特征根目录，包含train/test；默认data/features/{old,new}')
    args = parser.parse_args()
    t0 = time.time()
    print('extracting feats......')
    TrainData, TestData = getData(args.data_dir)
    getFeat(TrainData, TestData, args.params, args.feature_root)
    t1 = time.time()
    print("Features are extracted and saved.")
    print('total time is:%f' % (t1 - t0))
