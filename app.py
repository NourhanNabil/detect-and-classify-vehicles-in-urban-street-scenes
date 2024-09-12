import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import requests
from PIL import Image
from io import BytesIO

model = YOLO('yolov8.onnx')

st.title("Upload an Image for Vehicles Detection and Classification")

with st.form(key='upload_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    
    with col2:
        url = st.text_input("Or enter image URL")

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if uploaded_file is not None:
        extension = uploaded_file.name.split('.')[-1]
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    elif url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            extension = image.format.lower()
            image = np.array(image)

            if image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        except Exception as e:
            st.error(f"Error fetching image from URL: {e}")
            st.stop()

    else:
        st.error("Please upload an image file or provide a valid URL.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_file_path = os.path.join(tmpdirname, f"temp_image.{extension}")

        if uploaded_file:
            cv2.imwrite(temp_file_path, image)
        else:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(temp_file_path, image_bgr)

        result = model(temp_file_path, imgsz=480, conf=0.8)

        detection_plot = result[0].plot()

        detection_plot_rgb = cv2.cvtColor(detection_plot, cv2.COLOR_BGR2RGB)
        
        st.image(detection_plot_rgb, caption="Detected Image", use_column_width=True)

        st.success("Object detection completed!")
