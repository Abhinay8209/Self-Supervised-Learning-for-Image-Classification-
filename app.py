# ========== Imports ==========
import os
import cv2
import numpy as np
import gdown
import streamlit as st
from PIL import Image
import streamlit_authenticator as stauth

# ========== Authentication Setup ==========
# Already hashed passwords
users = {
    "usernames": {
        "john": {
            "name": "John Doe",
            "password": "$2b$12$XcTUE91Z52tTq7kNMDaQYOJuMbI1DHrlyEbtoXUuNBbN6iTOLd5yW"
        },
        "jane": {
            "name": "Jane Doe",
            "password": "$2b$12$H3YvTzTzBCxIMLMzPC8x7uRxV1J2FJx0zwmMLFKn0.N2AHx93CuCe"
        }
    }
}

# Setup authenticator
authenticator = stauth.Authenticate(
    users["usernames"],
    "cookie_name",
    "signature_key",
    cookie_expiry_days=1
)

# ========== Login Page ==========
name, authentication_status, username = authenticator.login('Login', 'main')

# ========== Main App ==========
if authentication_status is False:
    st.error('Username or password is incorrect ❌')

elif authentication_status is None:
    st.warning('Please enter your username and password 🛡️')

elif authentication_status:

    authenticator.logout('Logout', 'sidebar')
    st.sidebar.success(f"Welcome *{name}*! 🎉")

    # ========== Styling ==========
    st.markdown(
        """
        <h1 style='text-align: center; color: #4CAF50;'>Self-Supervised Learning: YOLO Object Detection 🖼️</h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ========== YOLO Model Setup ==========
    # Google Drive file ID for YOLO weights
    file_id = "1zT3hJatcXjfQuZBUJvXO7P2eXv1U_qGD"
    weights_path = "yolov3.weights"

    # Download the YOLO weights if not already present
    if not os.path.exists(weights_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, weights_path, quiet=False)

    # Load YOLO model
    yolo_config = "yolov3.cfg"
    coco_names = "coco.names"

    # Load COCO class labels
    with open(coco_names, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Load YOLO network
    net = cv2.dnn.readNet(weights_path, yolo_config)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # ========== Upload Image ==========
    uploaded_file = st.file_uploader("Upload an image 📷", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.success("✅ Image uploaded successfully!")

        try:
            # Read and prepare the image
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            height, width = img_array.shape[:2]

            # Create blob from image
            blob = cv2.dnn.blobFromImage(img_array, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward(output_layers)

            boxes, confidences, class_ids = [], [], []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(img_array, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img_array, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            st.image(img_array, caption="🔍 Detected Objects", use_column_width=True)

        except Exception as e:
            st.error(f"❌ Error occurred during detection: {e}")
