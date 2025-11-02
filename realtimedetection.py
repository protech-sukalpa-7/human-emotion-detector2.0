import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json, Sequential
from keras.layers import Conv2D, Dense, Flatten
import keras
from PIL import Image
import time

# Try importing streamlit-webrtc for live video
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

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
        "This application detects facial emotions using deep learning. "
        "Upload an image, capture from webcam, analyze video, or use live detection!"
    )
    st.sidebar.markdown("### Supported Emotions")
    st.sidebar.markdown(
        "- 😠 Angry\n- 🤢 Disgust\n- 😨 Fear\n- 😊 Happy\n- 😐 Neutral\n- 😢 Sad\n- 😲 Surprise"
    )
    
    # Show live video availability
    st.sidebar.markdown("---")
    if WEBRTC_AVAILABLE:
        st.sidebar.success("✅ Live Video: Available")
    else:
        st.sidebar.warning("⚠️ Live Video: Install streamlit-webrtc")

    # Load Models
    try:
        model = load_emotion_model()
        face_cascade = load_face_cascade()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

    # Mode Selection - Add Live Video if available
    modes = ["Upload Image", "Camera Capture", "Upload Video"]
    if WEBRTC_AVAILABLE:
        modes.insert(2, "Live Video Detection")
    
    mode = st.radio("Select Mode", modes, horizontal=True)

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

    # ---- Camera Capture Mode ----
    elif mode == "Camera Capture":
        st.subheader("📸 Camera Capture")
        
        st.info("👇 Click the button below to capture an image from your webcam")
        
        camera_photo = st.camera_input("Take a picture")
        
        if camera_photo is not None:
            file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                st.error("Failed to load image from camera.")
                return

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Captured Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

            with st.spinner("Detecting emotions..."):
                processed_image, results = detect_emotion(image.copy(), model, face_cascade)

            with col2:
                st.subheader("Detected Emotions")
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)

            if results:
                st.success(f"✅ Detected {len(results)} face(s)")
                for i, result in enumerate(results, 1):
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric(f"Face {i} - Emotion", result['emotion'])
                    with col_metric2:
                        st.metric("Confidence", f"{result['confidence']:.2f}%")
            else:
                st.warning("⚠️ No faces detected in the captured image")

    # ---- Live Video Detection Mode ----
    elif mode == "Live Video Detection":
        st.subheader("🎥 Live Video Emotion Detection")
        
        if not WEBRTC_AVAILABLE:
            st.error("❌ Live video requires the `streamlit-webrtc` library")
            st.code("pip install streamlit-webrtc", language="bash")
            st.info("After installing, add it to your `requirements.txt` for Streamlit Cloud deployment")
            return
        
        # Instructions
        st.info("📹 **Click START below and allow camera access when your browser prompts you**")
        
        # Video processor class
        class EmotionVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.model = load_emotion_model()
                self.face_cascade = load_face_cascade()
                self.frame_count = 0
                
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                
                # Process every frame for real-time detection
                processed_img, results = detect_emotion(img, self.model, self.face_cascade)
                
                self.frame_count += 1
                
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        
        # RTC Configuration with multiple STUN servers for better connectivity
        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
            ]
        })
        
        # Create webrtc streamer
        webrtc_ctx = webrtc_streamer(
            key="emotion-detection-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"min": 640, "ideal": 1280, "max": 1920},
                    "height": {"min": 480, "ideal": 720, "max": 1080},
                    "frameRate": {"ideal": 30, "max": 60},
                },
                "audio": False
            },
            async_processing=True,
        )
        
        st.divider()
        
        # Status display
        if webrtc_ctx.state.playing:
            st.success("✅ **Live Detection Active!** Your emotions are being detected in real-time.")
            
            # Tips for better detection
            with st.expander("💡 Tips for Better Detection"):
                st.markdown("""
                - **Face the camera directly** for best results
                - **Ensure good lighting** - avoid backlighting
                - **Stay within frame** - not too close or far
                - **Be expressive** - clear emotions are detected better
                - **Minimize movement** - reduces blur and improves accuracy
                """)
        else:
            st.info("⏸️ **Click START** above to begin live emotion detection")
        
        # Troubleshooting
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            ### Common Issues & Solutions:
            
            **1. START button does nothing**
            - Refresh the page and try again
            - Make sure you're on HTTPS (Streamlit Cloud provides this automatically)
            - Try a different browser (Chrome/Edge work best)
            
            **2. Camera permission denied**
            - Check browser settings: Settings → Privacy & Security → Camera
            - Make sure no other app is using the camera
            - Refresh the page and allow when prompted
            
            **3. Video is frozen or laggy**
            - Close other tabs/applications
            - Check your internet connection
            - Try lowering the video quality in your browser settings
            
            **4. No emotion detection showing**
            - Ensure good lighting
            - Face the camera directly
            - Move closer to the camera
            
            **5. For Local Development:**
            ```bash
            # Run on localhost for local testing
            streamlit run app.py --server.address localhost
            ```
            
            **6. For Streamlit Cloud Deployment:**
            - Add to `requirements.txt`:
              ```
              streamlit-webrtc
              opencv-python-headless
              keras
              tensorflow
              ```
            - Streamlit Cloud automatically provides HTTPS ✅
            """)

    # ---- Upload Video Mode ----
    elif mode == "Upload Video":
        st.subheader("🎥 Video Upload & Analysis")
        
        st.info("👇 Upload a pre-recorded video for frame-by-frame emotion analysis")
        
        video_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if video_file is not None:
            # Save uploaded video temporarily
            temp_video_path = "temp_video.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(video_file.read())
            
            # Process video
            st.subheader("🎬 Processing Video...")
            
            cap = cv2.VideoCapture(temp_video_path)
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            st.write(f"📹 **Video Info:** {total_frames} frames | {fps} FPS | {duration:.2f} seconds")
            
            # Processing options
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                process_every_n = st.slider("Process every N frames", 1, 30, 5, 
                                           help="Higher value = faster processing but less detailed")
            with col_opt2:
                max_frames_display = st.slider("Max frames to display", 5, 50, 20,
                                              help="Number of processed frames to show")
            
            if st.button("🚀 Start Video Analysis", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                processed_frames = []
                all_emotions = []
                frame_count = 0
                processed_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process only every Nth frame
                    if frame_count % process_every_n == 0:
                        processed_frame, results = detect_emotion(frame.copy(), model, face_cascade)
                        
                        if len(processed_frames) < max_frames_display:
                            processed_frames.append(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                        
                        # Collect emotions for statistics
                        for result in results:
                            all_emotions.append(result['emotion'])
                        
                        processed_count += 1
                    
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: Frame {frame_count}/{total_frames}")
                
                cap.release()
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ Processed {processed_count} frames!")
                
                # Display results
                if processed_frames:
                    st.subheader("📊 Processed Frames")
                    
                    # Display frames in grid
                    cols_per_row = 3
                    for i in range(0, len(processed_frames), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            if i + j < len(processed_frames):
                                with cols[j]:
                                    st.image(processed_frames[i + j], 
                                           caption=f"Frame {(i+j)*process_every_n}",
                                           use_container_width=True)
                
                # Emotion statistics
                if all_emotions:
                    st.subheader("📈 Emotion Statistics")
                    
                    from collections import Counter
                    emotion_counts = Counter(all_emotions)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Emotion Distribution:**")
                        for emotion, count in emotion_counts.most_common():
                            percentage = (count / len(all_emotions)) * 100
                            st.write(f"- {emotion}: {count} times ({percentage:.1f}%)")
                    
                    with col2:
                        st.write("**Dominant Emotion:**")
                        dominant = emotion_counts.most_common(1)[0]
                        st.metric("Most Common", dominant[0], f"{dominant[1]} occurrences")
                        
                        if len(emotion_counts) > 1:
                            second = emotion_counts.most_common(2)[1]
                            st.metric("Second Most", second[0], f"{second[1]} occurrences")
                else:
                    st.warning("⚠️ No faces detected in the video")
        
        else:
            st.markdown("""
            ### 📝 Instructions:
            
            1. **Record a video** using your phone or camera
            2. **Upload the video file** using the uploader above
            3. **Adjust processing settings** (optional)
            4. **Click "Start Video Analysis"** to process
            
            💡 **Tips:**
            - Keep video under 30 seconds for faster processing
            - Ensure good lighting and face the camera
            - Supported formats: MP4, AVI, MOV, MKV
            """)


# ------------------------------------------------------------
if __name__ == "__main__":
    main()
