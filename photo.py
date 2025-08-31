import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Press any key to capture")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Press any key to capture", frame)

    key = cv2.waitKey(1)  # Wait for a key press
    if key != -1:  # Any key press will trigger a capture
        filename = "captured_image.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")
        break  # Exit after capturing

cap.release()
cv2.destroyAllWindows()
