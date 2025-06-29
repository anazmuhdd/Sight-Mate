import cv2


import pyttsx3

from time import time
from ultralytics import YOLO

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 125)  # Adjust the voice speed (default is 200)
engine.say("Welcome")
engine.runAndWait()
engine.say("Performing detection")
engine.runAndWait()



# Open a connection to the USB camera (adjust the index based on your setup)
usb_camera_index = 0
cap = cv2.VideoCapture(usb_camera_index)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
# Check if the USB camera opened successfully
if not cap.isOpened():
    print(f"Error: Could not open USB camera at index {usb_camera_index}.")
    exit()

# Get the default frame dimensions (assuming 640x480)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Load the YOLO model
model_path = 'obj1.pt'
model = YOLO(model_path)

# Set the detection threshold
threshold = 0.85
# Variable to store the time of the last audio feedback
last_feedback_time = 0

# Define the gap duration (in seconds)
feedback_gap = 5

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

        # Draw bounding box if the score is above the threshold
        detected_objects.append(results.names[int(class_id)])  # Add detected object name to the list

    # Convert detected object names to audio feedback if 5 seconds have passed since the last feedback
    if detected_objects and time() - last_feedback_time >= feedback_gap:
        objects_text = ", ".join(detected_objects)
        try:
            engine.say(objects_text)
            engine.runAndWait()
            last_feedback_time = time()  # Update the last feedback time
        except Exception as e:
            print("Error:", e)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the USB camera and close all windows
cap.release()
cv2.destroyAllWindows()
