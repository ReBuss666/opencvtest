# Суротту корсотуу жана resize, rpotate, pixel modification сыяктуу негизги операцияларды аткаруу үчүн OpenCV жана Matplotlib колдонулат.
import cv2
import matplotlib.pyplot as plt


def run(image_path="example.jpg"):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Түс мейкиндиктери
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # Resize & Rotate
    resized_img = cv2.resize(img_rgb, (200, 200))
    rotated_img = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)

    # Pixel modification (кызыл квадрат тартуу)
    pixel_modified = img_rgb.copy()
    h, w, _ = pixel_modified.shape
    pixel_modified[h // 4 : h // 2, w // 4 : w // 2] = [255, 0, 0]

    titles = [
        "RGB",
        "HSB/HSV",
        "Greyscale",
        "Binary",
        "Resize (200x200)",
        "Rotate 90°",
        "Pixel Modified",
    ]
    images = [
        img_rgb,
        img_hsv,
        img_gray,
        img_binary,
        resized_img,
        rotated_img,
        pixel_modified,
    ]

    plt.figure(figsize=(14, 7))
    for i in range(7):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i], cmap="gray" if len(images[i].shape) == 2 else None)
        plt.title(titles[i])
        plt.axis("off")

    plt.suptitle("1. Basic Operations & Display")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
