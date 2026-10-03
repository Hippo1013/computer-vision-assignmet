#!/usr/bin/env python3
"""对照上游网络与完整 inference，核对 TF2 适配的数值。临时数组自动清理。"""
import argparse, json, os, subprocess, sys, tempfile, types
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def reference(folder):
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow.compat.v1 as tf

    tf.disable_eager_execution()
    sys.path.insert(0, str(ROOT / "MVSNet"))
    network = types.ModuleType("cnn_wrapper.network")
    source = (
        (ROOT / "MVSNet/cnn_wrapper/network.py")
        .read_text()
        .replace("import tensorflow as tf", "import tensorflow.compat.v1 as tf")
    )
    source = source.replace("tf.contrib.layers.l2_regularizer(1.0) if regularize else None", "None")
    source = source.replace("basestring", "str").replace("C / group_channel", "C // group_channel")
    exec(compile(source, "upstream_network_compat", "exec"), network.__dict__)
    sys.modules["cnn_wrapper.network"] = network
    homo = types.ModuleType("homography_warping")
    source = (
        (ROOT / "MVSNet/mvsnet/homography_warping.py")
        .read_text()
        .replace("import tensorflow as tf", "import tensorflow.compat.v1 as tf")
    )
    source = source.replace(
        "tf.contrib.image.transform(\n        input_image, homography_linear, interpolation='BILINEAR')",
        "tf.raw_ops.ImageProjectiveTransformV3(images=input_image, transforms=homography_linear, output_shape=tf.shape(input_image)[1:3], interpolation='BILINEAR', fill_mode='CONSTANT', fill_value=0.)",
    )
    exec(compile(source, "upstream_homography_compat", "exec"), homo.__dict__)
    sys.modules["homography_warping"] = homo
    module = types.ModuleType("upstream_model")
    source = (
        (ROOT / "MVSNet/mvsnet/model.py")
        .read_text()
        .replace("import tensorflow as tf", "import tensorflow.compat.v1 as tf")
        .replace("from convgru import ConvGRUCell", "")
    )
    exec(compile(source, "upstream_model_compat", "exec"), module.__dict__)
    tf.app.flags.DEFINE_integer("view_num", 3, "")
    tf.app.flags.DEFINE_integer("batch_size", 1, "")
    tf.app.flags.FLAGS(["reference"])
    arrays = np.load(folder / "input.npz")
    images = arrays["images"]
    cams = arrays["cams"]
    depths = arrays["depths"]
    image_tensor = tf.constant(images)
    cam_tensor = tf.constant(cams)
    depth, prob = module.inference(
        image_tensor,
        cam_tensor,
        len(depths),
        tf.constant([depths[0]]),
        tf.constant([depths[1] - depths[0]]),
    )
    saver = tf.train.Saver()
    with tf.Session(
        config=tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=1)
    ) as sess:
        saver.restore(sess, str(ROOT / "checkpoints/tf_model_blendedmvs/3DCNNs/model.ckpt-150000"))
        result = sess.run([depth, prob])
    np.savez(folder / "reference.npz", depth=result[0].squeeze(), confidence=result[1].squeeze())


def main():
    from mvsnet_local.io import load_view, CHECKPOINT, save_json
    from mvsnet_local.model import MVSNet

    model = MVSNet(CHECKPOINT, 4)
    views = [load_view(i, 64, 64) for i in [12, 10, 13]]
    images = np.stack([v[0] for v in views])[None]
    cams = np.zeros((1, 3, 2, 4, 4), np.float32)
    for i, v in enumerate(views):
        cams[0, i, 0] = v[3]
        cams[0, i, 1, :3, :3] = v[2]
    depths = 425 + np.arange(32, dtype=np.float32) * 2.65
    # 让两实现使用完全相同的浮点起点与步长。
    depths = depths[0] + np.arange(32, dtype=np.float32) * (depths[1] - depths[0])
    f = [model.features(v[0][None]) for v in views]
    d, p = model.predict(f, [(v[2], v[3]) for v in views], depths)
    with tempfile.TemporaryDirectory(prefix="mvsnet-parity-") as temp:
        folder = Path(temp)
        np.savez(folder / "input.npz", images=images, cams=cams, depths=depths)
        subprocess.run([sys.executable, __file__, "--reference", str(folder)], check=True)
        ref = np.load(folder / "reference.npz")
        result = {
            "reference": "YoYo000/MVSNet original inference + original CNN definitions; TF1 APIs replaced by TF2 compatibility APIs only",
            "input_shape": list(images.shape),
            "depth_samples": len(depths),
            "depth_max_abs_mm": float(np.max(np.abs(d - ref["depth"]))),
            "confidence_max_abs": float(np.max(np.abs(p - ref["confidence"]))),
        }
        np.testing.assert_allclose(d, ref["depth"], atol=0.02, rtol=1e-5)
        np.testing.assert_allclose(p, ref["confidence"], atol=5e-4, rtol=1e-3)
        result["passed"] = True
        out = ROOT / "outputs/summary"
        out.mkdir(exist_ok=True, parents=True)
        save_json(out / "model_parity.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    reference(args.reference) if args.reference else main()
