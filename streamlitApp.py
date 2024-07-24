import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import os
import platform
import pathlib

# Platform-specific path handling
plt = platform.system()
if plt == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Set up the page layout
st.set_page_config(page_title="ChromaticScan", page_icon="🌿")

st.title("ChromaticScan")

st.caption(
    "A ResNet 34-based Algorithm for Robust Plant Disease Detection with 99.2% Accuracy Across 39 Different Classes of Plant Leaf Images."
)

# Sidebar for navigation
with st.sidebar:
    img = Image.open("./Images/leaf.png")
    st.image(img)
    st.subheader("Navigation")
    page = st.radio("Go to", ["Home", "Prediction", "Charts"])

# Function to load and preprocess the image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Function to load the model
def load_model_file(model_path):
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    else:
        st.error("Model file not found. Please check the path and try again.")
        return None

# Function for Plant Disease Detection
def Plant_Disease_Detection(image):
    model = load_model_file("Plant_disease.h5")
    if model is None:
        return None, None, None

    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100  # Confidence level
    return predicted_class, confidence

# Home Page
if page == "Home":
    st.write(
        "Welcome to ChromaticScan, your AI-powered solution for detecting plant diseases. "
        "Use the sidebar to navigate to the prediction or charts sections."
    )

# Prediction Page
elif page == "Prediction":
    st.subheader("Upload an Image for Prediction")

    input_method = st.radio("Select Image Input Method", ["File Uploader", "Camera Input"], label_visibility="collapsed")

    if input_method == "File Uploader":
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            uploaded_file_img = load_image(uploaded_file)
            st.image(uploaded_file_img, caption="Uploaded Image", width=300)
            st.success("Image uploaded successfully!")

    elif input_method == "Camera Input":
        st.warning("Please allow access to your camera.")
        camera_image_file = st.camera_input("Click an Image")
        if camera_image_file is not None:
            camera_file_img = load_image(camera_image_file)
            st.image(camera_file_img, caption="Camera Input Image", width=300)
            st.success("Image clicked successfully!")

    # Button to trigger prediction
    submit = st.button(label="Submit Leaf Image")
    if submit:
        st.subheader("Output")
        if input_method == "File Uploader" and uploaded_file is not None:
            image = uploaded_file_img
        elif input_method == "Camera Input" and camera_image_file is not None:
            image = camera_file_img

        if image is not None:
            with st.spinner(text="This may take a moment..."):
                predicted_class, confidence = Plant_Disease_Detection(image)
                if predicted_class:
                    st.write(f"Prediction: {predicted_class}")
                    st.write(f"Description: {classes_and_descriptions.get(predicted_class, 'No description available.')}")
                    st.write(f"Confidence: {confidence:.2f}%")

                    # Prepare data for the table
                    recommendation = remedies.get(predicted_class, 'No recommendation available.')

                    data = {
                        "Details": ["Leaf Status", "Disease Name", "Recommendation", "Accuracy"],
                        "Values": ["Unhealthy" if predicted_class != "healthy" else "Healthy", 
                                   predicted_class.split('___')[1] if len(predicted_class.split('___')) > 1 else 'Healthy',
                                   recommendation,
                                   f"{confidence:.2f}%"]
                    }
                    df = pd.DataFrame(data)
                    st.table(df)
                else:
                    st.error("Error in prediction. Please try again.")
        else:
            st.warning("Please upload or capture an image first.")

# Charts Page
elif page == "Charts":
    st.subheader("Charts and Visualizations")
    st.write("This section can be used to display various charts and visualizations related to plant diseases and their detection. You can use libraries like Matplotlib, Seaborn, or Plotly to create interactive charts.")
