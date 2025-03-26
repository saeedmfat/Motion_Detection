import cv2
import time
import numpy as np

# Initialize video capture (0 for webcam, or path to video file)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create background subtractor (MOG2 with shadow detection)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,        # Number of frames to consider for background
    varThreshold=16,    # Sensitivity to changes (lower = more sensitive)
    detectShadows=True  # Detect and mark shadows (gray in the mask)
)

# Optional: Kernel for noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Motion detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Remove noise (morphological opening)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Find contours of moving objects
    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw bounding boxes around significant motion
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignore small contours (noise)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion_detected = True

    # Optional: Save frame if motion is detected
    if motion_detected:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"motion_{timestamp}.jpg", frame)

    # Display results
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Optional: Reset background model on 'r' key press
    if cv2.waitKey(1) & 0xFF == ord('r'):
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Release resources
cap.release()
cv2.destroyAllWindows()