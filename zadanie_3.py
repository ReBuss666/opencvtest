# Математикалык морфология
import cv2
import numpy as np
import matplotlib.pyplot as plt


def run(image_path="example.jpg"):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Сүрөт табылган жок: {image_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(bin_img, kernel, iterations=1)
    dilation = cv2.dilate(bin_img, kernel, iterations=1)
    opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    titles = ["Binary Original", "Erosion", "Dilation", "Opening", "Closing"]
    images = [bin_img, erosion, dilation, opening, closing]

    plt.figure(figsize=(14, 6))
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.suptitle("3. Mathematical Morphology")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
