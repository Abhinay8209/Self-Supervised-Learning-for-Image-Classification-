import os
import cv2
import numpy as np
import gdown
import streamlit as st
from PIL import Image

# Google Drive file ID for YOLO weights
file_id = "1zT3hJatcXjfQuZBUJvXO7P2eXv1U_qGD"
weights_path = "yolov3.weights"

# Download the YOLO weights if not already present
if not os.path.exists(weights_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, weights_path, quiet=False)

# YOLO config and COCO labels
yolo_config = "yolov3.cfg"
coco_names = "coco.names"

# Load COCO class labels
with open(coco_names, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load YOLO network
net = cv2.dnn.readNet(weights_path, yolo_config)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Streamlit UI
st.title("YOLO Object Detection with COCO Classes")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert image to OpenCV format
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(img_array, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Run forward pass
    outputs = net.forward(output_layers)

    # Initialize detection lists
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]  # Class scores
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Draw bounding boxes
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)  # Green color
        cv2.rectangle(img_array, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img_array, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display result
    st.image(img_array, caption="Detected Image", use_column_width=True)
