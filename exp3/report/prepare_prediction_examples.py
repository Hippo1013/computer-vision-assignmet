"""Export four test examples and replay their saved models without training."""
from pathlib import Path
import hashlib
import json
import pickle
import sys

import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent / "cv_homework1"
sys.path.insert(0, str(EXPERIMENT))
from extractFeat import rgb2gray
from extractFeat_TrafficSign import preprocess


def source(path):
    return {"path": str(path.relative_to(EXPERIMENT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def features(gray):
    return hog(gray, orientations=9, pixels_per_cell=(4, 4),
               cells_per_block=(4, 4), block_norm="L2-Hys",
               transform_sqrt=False, feature_vector=True)


def main():
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    metadata = {"selection": "Manually selected test examples; not a random sample or an accuracy estimate.",
                "index_convention": "zero-based saved test prediction row",
                "hog_config": [9, 4, 4], "examples": []}

    run = EXPERIMENT / "outputs/svm_tuning"
    selected = json.loads((run / "selection.json").read_text())["selected"]
    assert selected["id"] == "C0.2_squared_hinge" and selected["C"] == 0.2
    model_path = EXPERIMENT / selected["model_path"]
    assert source(model_path)["sha256"] == selected["model_sha256"]
    model = joblib.load(model_path)
    batch_path = EXPERIMENT / "cifar-10-batches-py/test_batch"
    with batch_path.open("rb") as stream:
        batch = pickle.load(stream, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    truth = np.asarray(batch[b"labels"])
    prediction_path = run / "predictions/C0.2_squared_hinge.npz"
    with np.load(prediction_path, allow_pickle=False) as saved:
        prediction = saved["test"]
    with np.load(run / "labels.npz", allow_pickle=False) as saved:
        assert np.array_equal(truth, saved["test"])
    assert float(np.mean(truth == prediction)) == selected["test_accuracy"] == 0.5635
    chinese = {3: "猫", 5: "狗", 8: "船"}
    for name, index, expected in (("cifar-correct", 1, True), ("cifar-incorrect", 0, False)):
        gray = rgb2gray(images[index]) / 255.0
        replay = int(model.predict(features(gray)[None])[0])
        assert replay == prediction[index]
        assert bool(truth[index] == replay) == expected
        output = figures / f"{name}.png"
        # Integer nearest-neighbor enlargement preserves the original 32x32 pixels.
        Image.fromarray(images[index]).resize((256, 256), Image.Resampling.NEAREST).save(output)
        metadata["examples"].append(dict(dataset="CIFAR-10", index=index, correct=expected,
            true_label=int(truth[index]), predicted_label=replay,
            true_chinese=chinese[int(truth[index])], predicted_chinese=chinese[replay],
            filename=batch[b"filenames"][index].decode(), svm_C=0.2,
            model=source(model_path), predictions=source(prediction_path), image_batch=source(batch_path),
            saved_model_replayed=True, output=output.name))

    run = EXPERIMENT / "outputs/traffic_hog_tuning_20261028_v1"
    protocol = json.loads((run / "protocol.json").read_text())
    assert json.loads((run / "selection.json").read_text())["selected"] == [9, 4, 4]
    result = json.loads((run / "o9_p4_b4/results.json").read_text())
    model_path = run / "o9_p4_b4/model.joblib"
    prediction_path = run / "o9_p4_b4/predictions.npz"
    assert source(model_path)["sha256"] == result["model_sha256"]
    assert source(prediction_path)["sha256"] == result["predictions_sha256"]
    model = joblib.load(model_path)
    assert model.C == result["svm_parameters"]["C"] == 1
    samples = {s["path"]: s for s in protocol["samples"]}
    # W1 is displayed as 警告标志, M7 as 人行横道标志; retain original codes in metadata.
    chinese = {9: "警告标志", 5: "人行横道标志"}
    with np.load(prediction_path, allow_pickle=False) as saved:
        assert int(np.sum(saved["test_true"] == saved["test_pred"])) == result["test_correct"] == 308
        for name, index, expected in (("traffic-correct", 319, True), ("traffic-incorrect", 323, False)):
            path = str(saved["test_paths"][index])
            image_path = EXPERIMENT / "Traffic_sign" / path
            true, pred = int(saved["test_true"][index]), int(saved["test_pred"][index])
            assert samples[path]["label"] == true
            assert source(image_path)["sha256"] == samples[path]["sha256"]
            assert path.split("/")[1] == protocol["class_names"][true]
            replay = int(model.predict(features(preprocess(image_path.read_bytes()))[None])[0])
            assert replay == pred and bool(true == pred) == expected
            output = figures / f"{name}.png"
            Image.open(image_path).convert("RGB").save(output)
            metadata["examples"].append(dict(dataset="Traffic_sign", index=index, correct=expected,
                true_label=true, predicted_label=pred,
                true_code=protocol["class_names"][true], predicted_code=protocol["class_names"][pred],
                true_chinese=chinese[true], predicted_chinese=chinese[pred], svm_C=1,
                image=source(image_path), model=source(model_path), predictions=source(prediction_path),
                saved_model_replayed=True, output=output.name))
    (figures / "prediction-examples-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print("Verified four image/label/prediction mappings and replayed both saved models; no training.")


if __name__ == "__main__":
    main()
