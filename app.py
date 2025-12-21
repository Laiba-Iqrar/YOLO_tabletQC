# =============================
# app.py — Tablet Quality Control (LIVE + IMAGE MODES, MOBILE FIX)
# =============================

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io, base64, warnings
from typing import Dict, Any

# --- YOLO + Live Video ---
from ultralytics import YOLO
import cv2
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# =============================
# Critical fixes
# =============================
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# =============================
# Page config
# =============================
st.set_page_config(
    page_title="Tablet Quality Control",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================
# CSS
# =============================
st.markdown("""
<style>
.stApp { background-color: #F5F5F0; }
h1, h2, h3 { color: #1E3A5F; font-family: 'Segoe UI', Arial; }
.square-image-container {
    width: 500px; height: 500px; background: #fff; border-radius: 8px;
    border: 2px solid #ccc; margin: 20px auto; display: flex;
    align-items: center; justify-content: center; overflow: hidden;
}
.square-image-container img { max-width: 100%; max-height: 100%; }
.defect-label { background:#E6EEF5; border:1px solid #1E3A5F; padding:10px;
    text-align:center; width:500px; margin:auto; font-weight:bold; }
.error-message { background:#F8D7DA; border:1px solid #DC3545; padding:10px;
    width:500px; margin:auto; text-align:center; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# =============================
# Utilities
# =============================

def pil_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)

# =============================
# IMAGE inference
# =============================

def run_model_inference(image: Image.Image, model_path: str) -> Dict[str, Any]:
    model = load_model(model_path)
    results = model.predict(image, conf=0.25, verbose=False)
    r = results[0]

    prediction = "No Defect"
    confidence = 1.0
    boxes_out = []

    if r.boxes is not None and len(r.boxes) > 0:
        best = r.boxes.conf.argmax()
        prediction = r.names[int(r.boxes.cls[best])]
        confidence = float(r.boxes.conf[best])

        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            boxes_out.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class": r.names[int(b.cls[0])],
                "confidence": float(b.conf[0])
            })

    return {
        "prediction": prediction,
        "confidence": confidence,
        "bboxes": boxes_out
    }

# =============================
# Annotation
# =============================

def create_annotated_image(image: Image.Image, result: Dict) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for b in result.get("bboxes", []):
        draw.rectangle([b['x1'], b['y1'], b['x2'], b['y2']], outline="#1E3A5F", width=3)
        label = f"{b['class']} {b['confidence']:.2f}"
        draw.text((b['x1'], max(0, b['y1'] - 12)), label, fill="#1E3A5F", font=font)

    return image

# =============================
# LIVE VIDEO PROCESSOR (MOBILE FIX)
# =============================

class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(img, conf=0.3, imgsz=640, verbose=False)
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# =============================
# Header
# =============================

st.title("In-Line Tablet Quality Control System")
st.markdown("### YOLO-Based Machine Vision Inspection")
st.markdown("---")

model_options = {
    "Model 1 (v4 100 Epochs)": "best_v4_100epochs.pt",
    "Model 2 (Final 305 Img)": "best_final_305 img_yolov8n_.pt",
    "Model 3 (Final Weights)": "Final_model_weights.pt"
}

selected_model = st.selectbox("Select Inspection Model", list(model_options.keys()))
model_path = model_options[selected_model]

# =============================
# Layout
# =============================

left, right = st.columns([2, 1])

# =============================
# RIGHT — INPUT MODE
# =============================

with right:
    st.subheader("Image Input")
    mode = st.radio("Mode", ["Upload Image", "Live Camera"])

    uploaded_image = None

    if mode == "Upload Image":
        file = st.file_uploader("Upload tablet image", type=["jpg", "png", "jpeg"])
        if file:
            uploaded_image = Image.open(file)

# =============================
# LEFT — OUTPUT
# =============================

with left:

    if mode == "Live Camera":
        st.subheader("📡 Live Camera Inspection")

        webrtc_streamer(
            key="tablet-live",
            video_processor_factory=lambda: YOLOVideoProcessor(model_path),
            media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
            video_html_attrs={"controls": False, "autoPlay": True},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True
        )

    elif uploaded_image:
        result = run_model_inference(uploaded_image, model_path)
        annotated = create_annotated_image(uploaded_image.copy(), result)
        img_html = pil_to_base64(annotated)

        st.markdown(f"""
        <div class="square-image-container">
            <img src="{img_html}">
        </div>
        <div class="defect-label">Detected: {result['prediction']}</div>
        """, unsafe_allow_html=True)

# =============================
# Footer
# =============================
st.markdown("---")
st.markdown("<center>Prototype HMI – Research Use Only</center>", unsafe_allow_html=True)
