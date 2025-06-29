# Make sure all necessary packages are installed before running
#pip install opencv-python ultralytics

import cv2
from ultralytics import YOLO
from time import time

# Open a connection to the USB camera (adjust the index based on your setup)
# Since this code is intended to run on your pc, I have set it in a way that it opens up your pc's webcam
usb_camera_index = 0
cap = cv2.VideoCapture(usb_camera_index)

# Check if the USB camera opened successfully
if not cap.isOpened():
    print(f"Error: Could not open USB camera at index {usb_camera_index}.")
    exit()

# Load the YOLO model
model_path = 'best.pt'  # Make sure this path is correct (Use obj1.pt for Object detection model and best.pt for currency recognition)
model = YOLO(model_path)

# Adjust the threshold (It's like a confidence measure 0.85 can be used as a default)
threshold = 0.85

while True:
    # Read a frame from the USB camera
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame from USB camera.")
        break

    # Perform inference on the frame
    results = model(frame)[0]

    detected_objects = []  # List to store detected object names

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # If the score is above the threshold, proceed
        if score > threshold:
            class_name = results.names[int(class_id)]
            detected_objects.append(class_name)

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} ({score:.2f})', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame with detection boxes
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the USB camera and close all windows
cap.release()
cv2.destroyAllWindows()
