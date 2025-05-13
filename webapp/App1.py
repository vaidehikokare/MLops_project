import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from keras.models import load_model

# Title
st.title("🎥 Violence Detection in Uploaded Videos")

# Load the trained Keras model
@st.cache_resource
def load_logged_model():
    model_path = r"models/trained31.h5"
    return load_model(model_path)

model = load_logged_model()

# Function to preprocess frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))  # resize to model input
    frame = frame.astype("float32") / 255.0  # normalize
    frame = np.expand_dims(frame, axis=0)  # add batch dimension
    return frame

# File uploader UI
uploaded_file = st.file_uploader("Upload a video file (MP4, AVI, MOV, MPEG4)", type=["mp4", "avi", "mov", "mpeg4"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    temp_video_path = tfile.name

    # Display video
    st.video(uploaded_file)

    # Load video using OpenCV
    cap = cv2.VideoCapture(temp_video_path)

    st.write("🔍 Analyzing video...")
    frame_count = 0
    violence_count = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Sample every 15th frame
        if total_frames % 15 == 0:
            preprocessed = preprocess_frame(frame)
            prediction = model.predict(preprocessed, verbose=0)[0][0]
            if prediction > 0.5:
                violence_count += 1
            frame_count += 1

    cap.release()
    os.unlink(temp_video_path)

    # Show results
    st.write(f"✅ Frames processed: {frame_count}")
    st.write(f"❗ Frames indicating violence: {violence_count}")

    if violence_count > 0:
        st.error("⚠️ Violence detected in the video.")
    else:
        st.success("✅ No violence detected in the video.")
