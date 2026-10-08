# Фурье өзгөртүүсү (Fourier Transformation)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def run(image_path="example.jpg"):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Сүрөт табылган жок: {image_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), 30, 1, -1)

    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.abs(np.fft.ifft2(f_ishift))

    titles = ["Original Gray", "Magnitude Spectrum", "Low-pass Filtered"]
    images = [gray, magnitude_spectrum, img_back]

    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.suptitle("7. Fourier Transformation")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
