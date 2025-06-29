import cv2
from ultralytics import YOLO

# Paths to YOLO models
model1_path = '../obj1.pt'
model2_path = '../best.pt'

# Load the first model initially
model = YOLO(model1_path)
current_model = 1

# Set detection threshold
threshold = 0.5

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Press 'm' to switch models | Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break

    # Run inference
    results = model(frame)[0]

    # Draw detections
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        if score >= threshold:
            label = results.names[int(class_id)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLO Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        if current_model == 1:
            model = YOLO(model2_path)
            current_model = 2
            print("Switched to best.pt")
        else:
            model = YOLO(model1_path)
            current_model = 1
            print("Switched to obj1.pt")

# Clean up
cap.release()
cv2.destroyAllWindows()
