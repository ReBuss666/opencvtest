import cv2 as cv

image_size = (640, 480)  # Set the desired image size (width, height)


def main():

    # Load an image from file
    image = cv.imread("example.jpg")
    print(image_size)
    # Check if the image was loaded successfully
    if image is None:
        print("Error: Could not load image.")
        return

    # Display the image in a window
    cv.namedWindow("Loaded Image", cv.WINDOW_AUTOSIZE | cv.WINDOW_GUI_NORMAL)
    cv.imshow("Loaded Image", image)

    # Wait for a key press indefinitely or for a specified amount of time
    cv.waitKey(0)

    # Destroy all OpenCV windows
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
