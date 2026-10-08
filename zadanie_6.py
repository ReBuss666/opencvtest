# Сүрөттү кысуу (Image Compression)
import cv2
import matplotlib.pyplot as plt


def run(image_path="example.jpg"):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Сүрөт табылган жок: {image_path}")

    _, enc_q90 = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    img_q90 = cv2.imdecode(enc_q90, 1)

    _, enc_q10 = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
    img_q10 = cv2.imdecode(enc_q10, 1)

    print("\n=== 6. Image Compression ===")
    print(f"JPEG Quality 90% Size: {len(enc_q90)} bytes")
    print(f"JPEG Quality 10% Size: {len(enc_q10)} bytes")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_q90, cv2.COLOR_BGR2RGB))
    plt.title(f"JPEG Quality 90 ({len(enc_q90)} bytes)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_q10, cv2.COLOR_BGR2RGB))
    plt.title(f"JPEG Quality 10 ({len(enc_q10)} bytes)")
    plt.axis("off")

    plt.suptitle("6. Image Compression")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
