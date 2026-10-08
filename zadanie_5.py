# Сүрөттү ченөө (Image Measurement)
import cv2
import matplotlib.pyplot as plt


def run(image_path="example.jpg"):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Сүрөт табылган жок: {image_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    measured_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    print("\n=== 5. Image Measurement ===")
    for i, cnt in enumerate(contours[:5]):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)

        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x
        cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y

        print(
            f"Object #{i + 1}: Area={area:.1f}, Perimeter={perimeter:.1f}, Centroid=({cX}, {cY}), BBox=({x}, {y}, {w}, {h})"
        )

        cv2.drawContours(measured_img, [cnt], -1, (0, 255, 0), 2)
        cv2.rectangle(measured_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(measured_img, (cX, cY), 4, (0, 0, 255), -1)

    plt.figure(figsize=(6, 6))
    plt.imshow(measured_img)
    plt.title("5. Measurement\nGreen: Contour, Blue: BBox, Red: Centroid")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
