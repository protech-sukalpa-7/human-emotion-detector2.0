import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json, Sequential
from keras.layers import Conv2D, Dense, Flatten
import keras
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import av
import asyncio
import nest_asyncio
nest_asyncio.apply()



# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Facial Emotion Detection",
    page_icon="😊",
    layout="wide"
)

# Register Sequential for backward compatibility
keras.utils.get_custom_objects().update({"Sequential": Sequential})


# ------------------------------------------------------------
# Load Model and Face Cascade
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

        cv2.putText(
            image, f"{emotion} ({confidence:.1f}%)",
            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 255, 0), 2
        )

    return image, results


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
def main():
    st.title("😊 Facial Emotion Detection System")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This application detects facial emotions in real-time using deep learning. "
        "Upload an image or use your webcam to detect emotions like Happy, Sad, Angry, and more!"
    )
    st.sidebar.markdown("### Supported Emotions")
    st.sidebar.markdown(
        "- 😠 Angry\n- 🤢 Disgust\n- 😨 Fear\n- 😊 Happy\n- 😐 Neutral\n- 😢 Sad\n- 😲 Surprise"
    )

    # Load Models
    try:
        model = load_emotion_model()
        face_cascade = load_face_cascade()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

    # Mode Selection
    mode = st.radio("Select Mode", ["Upload Image", "Webcam"], horizontal=True)

    # ---- Upload Image Mode ----
    if mode == "Upload Image":
        st.subheader("📤 Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                st.error("Failed to load image. Please try another file.")
                return

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

            with st.spinner("Detecting emotions..."):
                processed_image, results = detect_emotion(image.copy(), model, face_cascade)

            with col2:
                st.subheader("Detected Emotions")
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)

            if results:
                st.success(f"✅ Detected {len(results)} face(s)")
                for i, result in enumerate(results, 1):
                    st.metric(f"Face {i}", result['emotion'], f"{result['confidence']:.2f}% confidence")
            else:
                st.warning("⚠️ No faces detected in the image")

    # ---- Real-Time Webcam Mode ----
    elif mode == "Webcam":
        st.subheader("🎥 Live Emotion Detection")
        st.info("Allow access to your webcam to see live emotion detection in action.")

        class EmotionVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.model = load_emotion_model()
                self.face_cascade = load_face_cascade()

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                processed_img, _ = detect_emotion(img, self.model, self.face_cascade)
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

        webrtc_streamer(
            key="emotion",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionVideoProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
        )


# ------------------------------------------------------------
if __name__ == "__main__":
    main()
