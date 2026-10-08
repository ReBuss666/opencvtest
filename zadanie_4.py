# Сегментация (K-Means, Watershed)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def run(image_path="example.jpg"):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Сүрөт табылган жок: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # K-Means
    pixel_vals = img_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(pixel_vals)
    segmented_kmeans = (
        kmeans.cluster_centers_[kmeans.labels_].reshape(img_rgb.shape).astype(np.uint8)
    )

    # Watershed
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_bgr, markers)
    img_watershed = img_rgb.copy()
    img_watershed[markers == -1] = [255, 0, 0]

    titles = ["Original RGB", "K-Means Clustering", "Watershed Segmentation"]
    images = [img_rgb, segmented_kmeans, img_watershed]

    plt.figure(figsize=(12, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis("off")

    plt.suptitle("4. Image Segmentation")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
