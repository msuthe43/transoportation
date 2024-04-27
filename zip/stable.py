import cv2
import numpy as np
import csv

video_path = r'C:\Users\saver\3010project\transpo\zip\videoplayback.webm'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

scale_percent = 30  # Resize to 30% of the original size for speed improvement

# Define range for green color in HSV
lower_green = np.array([20, 40, 40])
upper_green = np.array([60, 255, 255])

# Skip the first 500 frames
for _ in range(500):
    ret, _ = cap.read()
    if not ret:
        print("Failed to skip the first 500 frames.")
        cap.release()
        exit()

ret, reference_frame = cap.read()
if not ret:
    print("Failed to capture the reference frame.")
    cap.release()
    exit()

# Resize the reference frame
width = int(reference_frame.shape[1] * scale_percent / 100)
height = int(reference_frame.shape[0] * scale_percent / 100)
reference_frame_resized = cv2.resize(reference_frame, (width, height))

# Filter the reference frame for green color and convert to grayscale for optical flow calculations
reference_hsv = cv2.cvtColor(reference_frame_resized, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(reference_hsv, lower_green, upper_green)
reference_green = cv2.bitwise_and(reference_frame_resized, reference_frame_resized, mask=mask)
reference_gray = cv2.cvtColor(reference_green, cv2.COLOR_BGR2GRAY)

with open('transformations2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'dx', 'dy'])

    frame_count = 501

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (width, height))

        # Apply green filter for optical flow calculations
        frame_hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame_hsv, lower_green, upper_green)
        frame_green = cv2.bitwise_and(frame_resized, frame_resized, mask=mask)
        frame_gray = cv2.cvtColor(frame_green, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(reference_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx, dy = flow[..., 0].mean(), flow[..., 1].mean()
        writer.writerow([frame_count, dx, dy])

        # Apply the transformation to the original resized frame
        transformation_matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
        stabilized_frame = cv2.warpAffine(frame_resized, transformation_matrix, (width, height))

        # Display the original resized frame and the stabilized frame
        cv2.imshow('Original Frame', frame_resized)
        cv2.imshow('Stabilized Frame', stabilized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

cap.release()
cv2.destroyAllWindows()
