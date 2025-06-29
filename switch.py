import cv2
import pyttsx3
from time import time
from ultralytics import YOLO
from gpiozero import Button

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 130) 
engine.say("Welcome")
engine.runAndWait()
engine.say("Please press the button to switch between the model")
engine.runAndWait()
engine.say("Object detection model")
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

# Load the initial YOLO model
initial_model_path = 'obj1.pt'
secondary_model_path = 'best.pt'

# Load the initial YOLO model
model = YOLO(initial_model_path)

# Set the detection threshold
threshold = 0.65

# Initialize the button
btn = Button(27)

# Variable to store the time of the last button press
last_button_press_time = 0

# Define the gap duration (in seconds) to avoid multiple model changes with a single button press
button_press_gap = 1

# Define the gap duration (in seconds)
last_feedback_time = 0
feedback_gap = 3.5

# Variable to track the current model
current_model_path = initial_model_path

# Function to switch between models
def switch_model():
    global model, current_model_path
    if current_model_path == initial_model_path:
        model = YOLO(secondary_model_path)
        current_model_path = secondary_model_path
    else:
        model = YOLO(initial_model_path)
        current_model_path = initial_model_path

while True:
    # Check if the button is pressed
    if btn.is_pressed:
        # Check if enough time has passed since the last button press
        if time() - last_button_press_time >= button_press_gap:
            # Switch the model
            switch_model()
            if current_model_path==initial_model_path:
             engine.say(f"Switched to object detection model")
             engine.runAndWait()
            else:
             engine.say(f"Switched to currency detection model")
            engine.runAndWait()
            # Update the time of the last button press
            last_button_press_time = time()

    # Read a frame from the USB camera
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame from USB camera.")
        break

    # Perform inference on the frame using the current YOLO model
    results = model(frame)[0]

    detected_objects = []  # List to store detected object names

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # Draw bounding box if the score is above the threshold
        if score > threshold:
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
