import streamlit as st
from PIL import Image
import torch
import os

# Load the model
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov11n", "custom", path="best.pt", force_reload=True)
    return model

# Initialize the model
model = load_model()

# Streamlit app
st.title("YOLOv11n Object Detection")
st.subheader("Upload an image to detect objects")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file to disk
    input_image_path = f"./{uploaded_file.name}"
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Run inference
    st.image(Image.open(input_image_path), caption="Uploaded Image", use_column_width=True)
    st.write("Running object detection...")

    results = model.predict(input_image_path, save=True, conf=0.5)
    detection_output = results.pandas().xyxy[0]  # Get the detections as a pandas DataFrame

    # Display detections
    st.write(f"Detection Results:\n{detection_output}")

    # Display the image with detections
    result_dir = "runs/detect"  # Default directory where YOLO saves results
    result_subdir = sorted(os.listdir(result_dir))[-1]  # Get the latest run directory
    result_image_path = os.path.join(result_dir, result_subdir, uploaded_file.name)
    st.image(Image.open(result_image_path), caption="Detected Objects", use_column_width=True)

    # Cleanup the saved input file
    os.remove(input_image_path)
