import os
import cv2
from ultralytics import YOLO

#  Setup paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, '..', 'best.pt')  # Adjust as needed
images_folder = os.path.join(BASE_DIR, 'test')
output_folder = os.path.join(BASE_DIR, 'inference_results')

# === Create output folder if it doesn't exist ===
os.makedirs(output_folder, exist_ok=True)

# === Load YOLO model ===
model = YOLO(model_path)
print("Model loaded")
print("Class names in model:", model.names)

# === Set detection threshold ===
threshold = 0.5

# === Run inference on each image ===
for filename in os.listdir(images_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(images_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping unreadable file: {filename}")
            continue

        # Inference
        results = model(image)[0]

        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            if score >= threshold:
                label = results.names[int(class_id)]
                print(f"Detected: {label} ({score:.2f}) in {filename}")

                # Draw bounding box and label
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, f"{label} {score:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save the result image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)
        print(f"Inference done: {filename} → saved to {output_path}")

print("🎉 All images processed.")
