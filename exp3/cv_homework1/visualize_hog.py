from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    img = cv2.imread('jinx.jpg')
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # TODO: 提取hog特征
    features, hog_vis = hog(
    gray_image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True,
    feature_vector=True
    )   

    # 可视化特征
    fig, ax = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title('input image')
    ax[1].imshow(hog_vis, cmap='hot')
    ax[1].set_title('visualization of HOG features')
    plt.show()

    from pathlib import Path

    # # 要求1：展示灰度图及其灰度直方图，并保存结果。
    # output_dir = Path(__file__).resolve().parent / 'outputs' / 'grayscale'
    # output_dir.mkdir(parents=True, exist_ok=True)
    # histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).ravel()
    # gray_fig, gray_axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    # gray_axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # gray_axes[0].set_title('Original image')
    # gray_axes[0].axis('off')
    # gray_axes[1].imshow(gray_image, cmap='gray', vmin=0, vmax=255)
    # gray_axes[1].set_title('Grayscale image')
    # gray_axes[1].axis('off')
    # gray_axes[2].bar(range(256), histogram, width=1, color='#087E88')
    # gray_axes[2].set_title('Grayscale histogram')
    # gray_axes[2].set_xlabel('Gray level')
    # gray_axes[2].set_ylabel('Pixel count')
    # gray_axes[2].set_xlim(-0.5, 255.5)
    # gray_axes[2].set_ylim(bottom=0)
    # gray_axes[2].grid(axis='y', alpha=0.2)
    # gray_fig.savefig(output_dir / 'grayscale_histogram.png', dpi=200)
    # cv2.imwrite(str(output_dir / 'jinx_gray.png'), gray_image)
    # plt.show()
