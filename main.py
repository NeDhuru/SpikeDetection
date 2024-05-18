import cv2
import numpy as np

def main():
    # Open a connection to the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Get frame dimensions
        height, width, _ = frame.shape

        # Define the square ROI
        roi_size = min(height, width) // 2  # Adjust the ROI size as needed
        roi_x1 = (width // 2) - (roi_size // 2)
        roi_y1 = (height // 2) - (roi_size // 2)
        roi_x2 = roi_x1 + roi_size
        roi_y2 = roi_y1 + roi_size

        # Draw the ROI rectangle on the frame
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

        # Crop the ROI from the frame
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Convert the ROI to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define the color range for detecting green
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Create a mask for the green color
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get the bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw the rectangle around the detected object
            cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Put the label 'spike' near the detected object
            cv2.putText(roi, "spike", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Replace the ROI in the frame with the processed ROI
        frame[roi_y1:roi_y2, roi_x1:roi_x2] = roi

        # Display the frame
        cv2.imshow('Frame', frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
