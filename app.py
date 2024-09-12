import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

model = YOLO('yolov8.onnx')

st.title("Upload an Image for Vehicles Detection and Classification")

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with tempfile.TemporaryDirectory() as tmpdirname:
        extension = uploaded_file.name.split('.')[-1]
        temp_file_path = os.path.join(tmpdirname, f"temp_image.{extension}")

        cv2.imwrite(temp_file_path, image)

        result = model(temp_file_path, imgsz=480, conf=0.8)

        detection_plot = result[0].plot()

        detection_plot_rgb = cv2.cvtColor(detection_plot, cv2.COLOR_BGR2RGB)
        
        st.image(detection_plot_rgb, caption="Detected Image", use_column_width=True)

        st.success("Object detection completed!")
