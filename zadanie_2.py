# Фильтрлөө жана чекти аныктоо (Edge Detection)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def run(image_path="example.jpg"):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Сүрөт табылган жок: {image_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Variance filter
    variance_filter = ndimage.generic_filter(gray, np.var, size=3)

    # 2. Range filter
    range_filter = ndimage.generic_filter(gray, lambda x: np.ptp(x), size=3)

    # 3. Robert's filter
    roberts_v = np.array([[1, 0], [0, -1]], dtype=float)
    roberts_h = np.array([[0, 1], [-1, 0]], dtype=float)
    roberts = np.sqrt(
        ndimage.convolve(gray.astype(float), roberts_v) ** 2
        + ndimage.convolve(gray.astype(float), roberts_h) ** 2
    )

    # 4. Kirsch's template filter
    kirsch_masks = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
    ]
    kirsch = np.max(
        [ndimage.convolve(gray.astype(float), k) for k in kirsch_masks], axis=0
    )

    # 5. Prewitt's gradient filter
    prewitt_x = cv2.filter2D(gray, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    prewitt_y = cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
    prewitt = cv2.addWeighted(np.abs(prewitt_x), 0.5, np.abs(prewitt_y), 0.5, 0)

    # 6. Sobel's gradient filter
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)

    # 7. Gaussian + Canny thinning
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
    canny = cv2.Canny(gaussian_blur, 50, 150)

    titles = [
        "Variance",
        "Range",
        "Roberts",
        "Kirsch",
        "Prewitt",
        "Sobel",
        "Gaussian + Canny",
    ]
    images = [variance_filter, range_filter, roberts, kirsch, prewitt, sobel, canny]

    plt.figure(figsize=(15, 8))
    for i in range(7):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.suptitle("2. Filters & Edge Detection")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
