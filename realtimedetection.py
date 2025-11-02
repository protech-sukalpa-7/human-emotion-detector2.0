import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json, Sequential
import keras
from PIL import Image
import platform

# Safe import for live video dependencies
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
    import av
    WEBRTC_AVAILABLE = True
except Exception as e:
    WEBRTC_AVAILABLE = False

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(page_title="Facial Emotion Detection", page_icon="😊", layout="wide")

keras.utils.get_custom_objects().update({"Sequential": Sequential})


# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------
@st.cache_resource
def load_emotion_model():
    with open("facialemotionmodel.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("facialemotionmodel.h5")
    return model


@st.cache_resource
def load_face_cascade():
    paths = [
        'haarcascade_frontalface_default.xml',
        cv2.__file__[:-11] + 'data/haarcascade_frontalface_default.xml',
    ]
    for haar_file in paths:
        cascade = cv2.CascadeClassifier(haar_file)
        if not cascade.empty():
            return cascade
    raise FileNotFoundError("Could not find haarcascade_frontalface_default.xml.")


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


def detect_emotion(image, model, face_cascade):
    labels = {
        0: 'Angry', 1: 'Disgust', 2: 'Fear',
        3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'
    }
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []
    for (x, y, w, h) in faces:
        face_image = gray[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_image = cv2.resize(face_image, (48, 48))
        img = extract_features(face_image)

        pred = model.predict(img, verbose=0)
        emotion = labels[pred.argmax()]
        confidence = np.max(pred) * 100

        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'position': (x, y, w, h)
        })

        cv2.putText(image, f"{emotion} ({confidence:.1f}%)",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    return image, results


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
def main():
    st.title("😊 Facial Emotion Detection System")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info("Detect emotions from faces in images, webcam or video using deep learning.")
    st.sidebar.markdown("### Supported Emotions")
    st.sidebar.markdown(
        "- 😠 Angry\n- 🤢 Disgust\n- 😨 Fear\n- 😊 Happy\n- 😐 Neutral\n- 😢 Sad\n- 😲 Surprise"
    )

    # Check WebRTC support
    if WEBRTC_AVAILABLE:
        st.sidebar.success("✅ Live Video Detection Available")
    else:
        st.sidebar.warning("⚠️ Live Video Disabled (requires `streamlit-webrtc` + `av`)")

    # Load models
    try:
        model = load_emotion_model()
        face_cascade = load_face_cascade()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Mode selector
    modes = ["Upload Image", "Camera Capture", "Upload Video"]
    if WEBRTC_AVAILABLE and platform.system() != "Linux":
        modes.insert(2, "Live Video Detection")

    mode = st.radio("Select Mode", modes, horizontal=True)

    # Upload Image
    if mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
                with st.spinner("Detecting emotions..."):
                    processed_image, results = detect_emotion(image, model, face_cascade)
                with col2:
                    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB),
                             caption="Detected", use_container_width=True)
                if results:
                    for i, result in enumerate(results):
                        st.metric(f"Face {i+1}", result['emotion'], f"{result['confidence']:.2f}%")

    # Camera Capture
    elif mode == "Camera Capture":
        camera_photo = st.camera_input("Take a photo")
        if camera_photo:
            file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is not None:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Captured", use_container_width=True)
                processed_image, results = detect_emotion(image, model, face_cascade)
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Detected", use_container_width=True)
                if results:
                    for i, result in enumerate(results):
                        st.metric(f"Face {i+1}", result['emotion'], f"{result['confidence']:.2f}%")

    # Live Video Detection (safe version)
    elif mode == "Live Video Detection" and WEBRTC_AVAILABLE:
        st.subheader("🎥 Live Emotion Detection")

        class EmotionProcessor(VideoProcessorBase):
            def __init__(self):
                self.model = load_emotion_model()
                self.face_cascade = load_face_cascade()

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img, _ = detect_emotion(img, self.model, self.face_cascade)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        webrtc_streamer(
            key="emotion-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=EmotionProcessor,
            async_processing=True
        )

    # Upload Video Mode (unchanged)
    elif mode == "Upload Video":
        video_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
        if video_file:
            temp_path = "uploaded_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(video_file.read())

            cap = cv2.VideoCapture(temp_path)
            frames_processed = 0
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame, _ = detect_emotion(frame, model, face_cascade)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                frames_processed += 1

            cap.release()
            st.success(f"Processed {frames_processed} frames")


# ------------------------------------------------------------
if __name__ == "__main__":
    main()
