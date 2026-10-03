"""DTU 输入与 PFM / PLY 输出，路径不依赖终端所在目录。"""

from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "scan9"
CHECKPOINT = ROOT / "checkpoints/tf_model_blendedmvs/3DCNNs/model.ckpt-150000"


def read_pairs(path=DATA / "pair.txt"):
    lines = Path(path).read_text().splitlines()
    result = {}
    for i in range(int(lines[0])):
        tokens = lines[2 * i + 2].split()
        result[int(lines[2 * i + 1])] = [int(x) for x in tokens[1::2]]
    return result


def read_camera(path):
    lines = Path(path).read_text().splitlines()
    extrinsic = np.array([[float(x) for x in l.split()] for l in lines[1:5]], np.float32)
    intrinsic = np.array([[float(x) for x in l.split()] for l in lines[7:10]], np.float32)
    depths = list(map(float, lines[11].split()))
    return intrinsic, extrinsic, depths[0], depths[1]


def load_view(index, width=1152, height=864, data=DATA):
    """与上游一致：BGR、等比缩放、中心裁剪、四分之一内参。"""
    bgr = cv2.imread(str(data / "images" / f"{index:08d}.jpg"))
    if bgr is None:
        raise FileNotFoundError(index)
    K, E, start, interval = read_camera(data / "cams" / f"{index:08d}_cam.txt")
    scale = max(width / bgr.shape[1], height / bgr.shape[0])
    bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    K[:2] *= scale
    y = int(np.ceil((bgr.shape[0] - height) / 2))
    x = int(np.ceil((bgr.shape[1] - width) / 2))
    bgr = bgr[y : y + height, x : x + width]
    assert bgr.shape[:2] == (height, width)
    K[0, 2] -= x
    K[1, 2] -= y
    K[:2] *= 0.25
    rgb = cv2.cvtColor(cv2.resize(bgr, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB)
    image = bgr.astype(np.float32)
    image = (image - image.mean((0, 1), keepdims=True)) / (image.std((0, 1), keepdims=True) + 1e-8)
    return image, rgb, K, E, start, interval


def write_pfm(path, data):
    data = np.asarray(data, dtype="<f4")
    with open(path, "wb") as f:
        f.write(f"Pf\n{data.shape[1]} {data.shape[0]}\n-1.0\n".encode())
        np.flipud(data).tofile(f)


def read_pfm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"Pf"
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        a = np.fromfile(f, dtype="<f4" if scale < 0 else ">f4")
    return np.flipud(a.reshape(h, w)).copy()


def write_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, np.float32)
    rgb = np.asarray(rgb, np.uint8)
    assert xyz.shape == rgb.shape and xyz.shape[1] == 3 and np.isfinite(xyz).all()
    vertices = np.empty(
        len(xyz),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    for i, key in enumerate(("x", "y", "z")):
        vertices[key] = xyz[:, i]
    for i, key in enumerate(("red", "green", "blue")):
        vertices[key] = rgb[:, i]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(
            (
                "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(xyz)
                + "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            ).encode()
        )
        vertices.tofile(f)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
