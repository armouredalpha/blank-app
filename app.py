import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# Page config
st.set_page_config(
    page_title="Hand Gesture Detection Workshop",
    page_icon="🖐️",
    layout="wide"
)

# Title
st.title("🖐️ Live Hand Gesture Detection Workshop")
st.markdown("---")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# =========================================================
# STUDENT EDIT ZONE
# =========================================================
def detect_gesture(hand_landmarks):
    """
    Students: Add your gesture detection logic here!
    
    Landmarks reference:
    - 0: Wrist
    - 4: Thumb tip
    - 8: Index tip
    - 12: Middle tip
    - 16: Ring tip
    - 20: Pinky tip
    
    Each landmark has .x, .y, .z (0-1 range)
    """
    
    # Count extended fingers
    extended = 0
    
    # Thumb (check x-axis)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        extended += 1
    
    # Other fingers (check y-axis)
    finger_tips = [8, 12, 16, 20]
    finger_bases = [6, 10, 14, 18]
    
    for tip, base in zip(finger_tips, finger_bases):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            extended += 1
    
    # Return gesture name
    gesture_map = {
        0: "✊ Fist",
        1: "☝️ One Finger",
        2: "✌️ Peace Sign",
        3: "🤟 Three Fingers",
        4: "🤚 Four Fingers",
        5: "🖐️ Open Hand"
    }
    
    return gesture_map.get(extended, "Unknown")

# =========================================================
# VIDEO PROCESSOR
# =========================================================
class HandDetector(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Flip for selfie view
        img = cv2.flip(img, 1)
        
        # Convert to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.hands.process(rgb)
        
        # Draw landmarks and detect gestures
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=2)
                )
                
                # Detect gesture
                gesture = detect_gesture(hand_landmarks)
                
                # Display gesture
                cv2.putText(
                    img,
                    f"Hand {idx+1}: {gesture}",
                    (10, 50 + idx*60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3
                )
        else:
            cv2.putText(
                img,
                "Show Your Hand",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )
        
        return img

# =========================================================
# STREAMLIT UI
# =========================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Detection")
    
    # WebRTC configuration
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    webrtc_streamer(
        key="hand-detection",
        video_transformer_factory=HandDetector,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("📝 Instructions")
    st.markdown("""
    ### How to Use:
    1. Click **START** to begin
    2. Allow camera permissions
    3. Show your hand to camera
    4. See gesture detection!
    
    ### Tips:
    - Good lighting helps
    - Keep hand visible
    - Try different gestures
    
    ### Student Task:
    Modify `detect_gesture()` to recognize:
    - 👍 Thumbs up
    - 👎 Thumbs down
    - 👌 OK sign
    - 🤘 Rock on
    - Custom gestures!
    """)
    
    st.markdown("---")
    st.info("💡 Edit the code to add your own gestures!")

st.markdown("---")
st.markdown("### 🎓 Workshop by [Your Name]")
