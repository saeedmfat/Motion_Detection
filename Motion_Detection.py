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