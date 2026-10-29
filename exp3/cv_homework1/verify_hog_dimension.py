"""Verify the HOG descriptor dimension required by experiment item 3."""
from pathlib import Path
import json
import sys

import cv2
import numpy as np
import skimage
from skimage.feature import hog


def main():
    root = Path(__file__).resolve().parent
    image_path = root / "jinx.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.shape != (432, 768):
        raise ValueError(f"Expected image shape (432, 768), got {gray.shape}")

    parameters = dict(orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(16, 16), block_norm="L2-Hys")
    blocks = hog(gray, **parameters, feature_vector=False)
    features = hog(gray, **parameters, feature_vector=True)
    cells = tuple(size // cell for size, cell in zip(gray.shape, parameters["pixels_per_cell"]))
    block_grid = tuple(count - block + 1 for count, block in zip(cells, parameters["cells_per_block"]))
    expected_shape = (*block_grid, *parameters["cells_per_block"], parameters["orientations"])
    expected_dimension = int(np.prod(expected_shape))
    if blocks.shape != expected_shape or features.shape != (expected_dimension,):
        raise AssertionError("Computed HOG shape does not match the dimension derivation")
    if not np.isfinite(features).all():
        raise AssertionError("The HOG vector contains non-finite values")
    np.testing.assert_array_equal(features, blocks.ravel())

    result = dict(
        image="jinx.jpg", image_shape=list(gray.shape), parameters=parameters,
        cell_grid=list(cells), block_grid=list(block_grid),
        per_block_dimension=int(np.prod(parameters["cells_per_block"]) * parameters["orientations"]),
        expected_dimension=expected_dimension, block_array_shape=list(blocks.shape),
        feature_vector_shape=list(features.shape), flatten_matches=True, all_finite=True,
        python_executable=sys.executable, python_version=sys.version.split()[0],
        skimage_version=skimage.__version__, numpy_version=np.__version__, opencv_version=cv2.__version__,
    )
    destination = root / "outputs/hog_dimension"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
