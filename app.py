import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import os

# Set Streamlit page config
st.set_page_config(
    page_title="YOLOv11 Item Detection",
    page_icon="🛒",
    layout="wide"
)

# Title and description
st.title("YOLOv11 Item Detection App")
st.write("Upload an image to detect items using the trained YOLOv11 model.")

# Sidebar
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Load YOLO model
@st.cache_resource  # Cache the model for better performance
def load_model():
    model = YOLO("best.pt")  # Replace 'best.pt' with the path to your model
    return model

model = load_model()

# Process uploaded image
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded file to a temporary location
    temp_file_path = f"temp_{uploaded_file.name}"
    image.save(temp_file_path)

    # Perform prediction
    st.write("Detecting items...")
    results = model.predict(source=temp_file_path, conf=0.5, save=False)

    # Display predictions
    predictions = results[0]
    st.write(f"Detected objects: {len(predictions.boxes)}")

    # Draw boxes on the image
    for box in predictions.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0]
        class_name = results.names[int(box.cls[0])]
        st.write(f"Class: {class_name}, Confidence: {confidence:.2f}")

    # Display annotated image
    annotated_image_path = os.path.join("runs", "predict", "image.jpg")  # Modify if needed
    if os.path.exists(annotated_image_path):
        st.image(annotated_image_path, caption="Annotated Image", use_column_width=True)

    # Cleanup
    os.remove(temp_file_path)

# Instructions if no file is uploaded
else:
    st.write("Upload an image to get started.")
