import cv2
import numpy as np
from tkinter import Tk, filedialog

# Load YOLO model
yolo_config = "yolov3.cfg"
yolo_weights = "yolov3.weights"
coco_names = "coco.names"

# Load COCO class labels
with open(coco_names, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load YOLO network
net = cv2.dnn.readNet(yolo_weights, yolo_config)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Hide the root Tkinter window
root = Tk()
root.withdraw()

while True:  # Keep running until manually stopped
    # Open file dialog to select an image
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])

    if not image_path:  # If no image is selected, exit the loop
        print("No image selected. Exiting...")
        break

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read the image. Check the file path!")
        continue  # Skip and allow another selection

    height, width, channels = image.shape

    # Convert image to YOLO input format
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Run forward pass and get predictions
    outputs = net.forward(output_layers)

    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []

    # Process each detected object
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # Class scores
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                # Get bounding box coordinates
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to reduce overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Draw bounding boxes and labels
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)  # Green for humans/objects
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the result
    cv2.imshow("Object Detection", image)
    
    # Wait for a key press, close window on 'q' press
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()  # Close the window before selecting a new image

print("Program terminated.")
