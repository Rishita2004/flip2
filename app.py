import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
from collections import Counter

# Streamlit page configuration
st.set_page_config(
    page_title="YOLO Item Counter",
    page_icon="📦",
    layout="wide"
)

# Title and description
st.title("YOLO Item Counter App")
st.write("Upload an image to detect and count items using the trained YOLO model.")

# Load YOLO model
@st.cache_resource  # Cache the model to avoid reloading every time
def load_model():
    model = YOLO("best.pt")  # Ensure 'best.pt' is in the same directory as this script
    return model

model = load_model()

# Sidebar for uploading an image
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process the uploaded image
if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image temporarily
    temp_file_path = "uploaded_image.jpg"
    image.save(temp_file_path)

    # Perform predictions
    st.write("Detecting and counting items...")
    results = model.predict(source=temp_file_path, conf=0.5, save=False)  # Perform inference

    # Parse results to count objects
    detected_classes = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Class index
            class_name = result.names[class_id]  # Class name
            detected_classes.append(class_name)

    # Count occurrences of each class
    class_counts = Counter(detected_classes)

    # Display the counts
    st.write("### Item Counts:")
    for class_name, count in class_counts.items():
        st.write(f"- **{class_name}**: {count}")

    # Remove the temporary image file
    os.remove(temp_file_path)

else:
    st.write("Please upload an image to start detection.")

# Footer
st.sidebar.markdown("Developed with 💡 by [Your Name]")
