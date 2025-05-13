import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import tempfile
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Streamlit UI Setup
st.set_page_config(page_title="Violence Detection System", page_icon="🔍", layout="wide")

MODEL_PATH = "models/trained31.h5"

@st.cache_resource
def load_trained_model(model_path):
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model(MODEL_PATH)

IMG_SIZE = (225, 225)  # FIXED: Match model's expected input shape

def preprocess_frame(frame):
    frame = cv2.resize(frame, IMG_SIZE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)  # shape: (1, 225, 225, 3)

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    MAX_FRAMES = 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or len(frames) >= MAX_FRAMES:
            break
        frames.append(preprocess_frame(frame)[0])

    cap.release()
    
    if len(frames) == 0:
        return None

    frames = np.array(frames)
    frames = np.mean(frames, axis=0)  # average frame
    return np.expand_dims(frames, axis=0)

def send_email(subject, body, to_email):
    try:
        from_email = "kokarevaidehi2@gmail.com"
        password = "nmgf dtms aalx ctwe"
        
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()

        st.success("✅ Email sent successfully!")
    except Exception as e:
        st.error(f"Error sending email: {e}")

# Streamlit App UI
st.title("Violence Detection System")
st.sidebar.header("⚡ Choose Mode")
mode = st.sidebar.radio("Select Video Mode:", ["📂 Upload Video", "🎥 Use Webcam"])

if mode == "📂 Upload Video":
    st.subheader("📂 Upload a Video File")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_video_path = temp_file.name

        st.video(temp_video_path)
        st.write("🔄 **Processing video...**")

        processed_frame = preprocess_video(temp_video_path)

        if processed_frame is None:
            st.error("⚠️ No valid frames extracted from the video.")
        else:
            predictions = model.predict(processed_frame)
            confidence = predictions[0][0]
            st.write(f"📊 Raw Model Output: {confidence:.4f}")

            predicted_label = "✅ **No Violence**" if confidence > 0.5 else "🚨 **Violence Detected!**"
            st.subheader(f"Prediction: {predicted_label} (Confidence: {confidence:.2f})")

            if confidence <= 0.5:
                send_email(
                    subject="🚨 Violence Detected in Uploaded Video",
                    body=f"Violence detected with confidence: {confidence:.2f}",
                    to_email="vaidehikokare35@gmail.com"
                )

        os.remove(temp_video_path)

elif mode == "🎥 Use Webcam":
    st.subheader("🎥 **Live Video Feed**")

    col1, col2 = st.columns(2)
    with col1:
        start_video = st.button("▶ **Start Video**")
    with col2:
        stop_video = st.button("⏹ **Stop Video**")

    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "pred_buffer" not in st.session_state:
        st.session_state.pred_buffer = []
    if "violence_start_time" not in st.session_state:
        st.session_state.violence_start_time = None
    if "alert_sent" not in st.session_state:
        st.session_state.alert_sent = False

    if start_video:
        st.session_state.recording = True
        st.session_state.pred_buffer = []
        st.session_state.violence_start_time = None
        st.session_state.alert_sent = False

    if stop_video:
        st.session_state.recording = False

    FRAME_WINDOW = st.empty()
    alert_placeholder = st.empty()

    if st.session_state.recording:
        cap = cv2.VideoCapture(0)

        while st.session_state.recording:
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Could not access the webcam.")
                break

            FRAME_WINDOW.image(frame, channels="BGR", caption="Live Video")

            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)[0][0]

            st.session_state.pred_buffer.append(prediction)
            if len(st.session_state.pred_buffer) > 5:
                st.session_state.pred_buffer.pop(0)

            avg_prediction = np.mean(st.session_state.pred_buffer)

            if avg_prediction < 0.5:
                alert_placeholder.error(f"🚨 Violence Detected! (Confidence: {avg_prediction:.2f})")

                if st.session_state.violence_start_time is None:
                    st.session_state.violence_start_time = time.time()
                elif not st.session_state.alert_sent and (time.time() - st.session_state.violence_start_time >= 10):
                    send_email(
                        subject="🚨 Prolonged Violence Detected in Live Webcam Feed",
                        body=f"Violence detected continuously for over 10 seconds.\nConfidence: {avg_prediction:.2f}",
                        to_email="vaidehikokare35@gmail.com"
                    )
                    st.session_state.alert_sent = True
            else:
                alert_placeholder.success(f"✅ No Violence (Confidence: {avg_prediction:.2f})")
                st.session_state.violence_start_time = None
                st.session_state.alert_sent = False

        cap.release()
        cv2.destroyAllWindows()
        FRAME_WINDOW.empty()
        alert_placeholder.empty()
        st.success("✅ Video stopped.")
