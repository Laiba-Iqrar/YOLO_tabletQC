# =============================
# app.py — Tablet Quality Control (LIVE + IMAGE MODES, DROIDCAM)
# =============================

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io, base64, warnings, time
from typing import Dict, Any

# --- YOLO ---
from ultralytics import YOLO
import cv2
import numpy as np

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
# CSS (unchanged)
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
model = load_model(model_path)

# =============================
# Layout
# =============================

left, right = st.columns([2, 1])

# =============================
# RIGHT — INPUT MODE
# =============================

with right:
    st.subheader("Image Input")
    mode = st.radio("Mode", ["Upload Image", "Live Camera (DroidCam)"])

    uploaded_image = None

    if mode == "Upload Image":
        file = st.file_uploader("Upload tablet image", type=["jpg", "png", "jpeg"])
        if file:
            uploaded_image = Image.open(file)

# =============================
# LEFT — OUTPUT
# =============================

with left:

    if mode == "Live Camera (DroidCam)":
        st.subheader("📡 Live Camera Inspection (DroidCam)")

        start = st.button("▶ Start Camera")
        stop = st.button("⏹ Stop Camera")

        frame_placeholder = st.empty()
        status = st.empty()

        CAMERA_INDEX = 4   # change to 1 if needed

        if start:
            cap = cv2.VideoCapture(CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

            if not cap.isOpened():
                st.error("❌ DroidCam not detected. Make sure it is running.")
            else:
                status.success("✅ DroidCam connected")

                while cap.isOpened():
                    if stop:
                        break

                    ret, frame = cap.read()
                    if not ret:
                        status.error("⚠ Camera frame not received")
                        break

                    # Zoom to avoid ultra-wide lens
                    frame = cv2.resize(frame, None, fx=1.4, fy=1.4)

                    results = model(frame, conf=0.3, verbose=False)
                    annotated = results[0].plot()

                    frame_placeholder.image(
                        annotated,
                        channels="BGR",
                        use_container_width=True
                    )

                    time.sleep(0.02)

                cap.release()
                status.info("⏹ Camera stopped")

    elif uploaded_image:
        result = run_model_inference(uploaded_image, model_path)
        annotated = create_annotated_image(uploaded_image.copy(), result)
        img_html = pil_to_base64(annotated)

        st.markdown(f"""
        <div class=\"square-image-container\">
            <img src=\"{img_html}\">
        </div>
        <div class=\"defect-label\">Detected: {result['prediction']}</div>
        """, unsafe_allow_html=True)

# =============================
# Footer
# =============================
st.markdown("---")
st.markdown("<center>Prototype HMI – Research Use Only</center>", unsafe_allow_html=True)