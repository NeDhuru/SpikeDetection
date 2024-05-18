import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        height, width, _ = frame.shape

        roi_size = min(height, width) // 2  # Adjust the ROI size as needed
        roi_x1 = (width // 2) - (roi_size // 2)
        roi_y1 = (height // 2) - (roi_size // 2)
        roi_x2 = roi_x1 + roi_size
        roi_y2 = roi_y1 + roi_size

        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(roi, "spike", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame[roi_y1:roi_y2, roi_x1:roi_x2] = roi
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
