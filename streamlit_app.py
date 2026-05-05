import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("traffic_sign_model.h5")

classes = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End of speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing > 3.5 tons",
    "Right-of-way", "Priority road", "Yield", "Stop",
    "No vehicles", "No trucks", "No entry",
    "General caution", "Curve left", "Curve right",
    "Double curve", "Bumpy road", "Slippery road",
    "Road narrows", "Road work", "Traffic signals",
    "Pedestrians", "Children crossing", "Bicycles crossing",
    "Ice/Snow", "Wild animals", "End of all limits",
    "Turn right", "Turn left", "Go straight",
    "Straight or right", "Straight or left", "Keep right",
    "Keep left", "Roundabout", "End no passing",
    "End no passing >3.5t"
]

st.set_page_config(page_title="Traffic Sign Recognition", layout="wide")

# ----------- CSS -----------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #38bdf8;
}
.tagline {
    text-align: center;
    font-size: 22px;
    color: #cbd5e1;
    margin-bottom: 30px;
}
.upload-box {
    border: 2px dashed #38bdf8;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;

}

/* BIG PROFESSIONAL BUTTON */
.stButton>button {
    width: 100%;
    padding: 18px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    background: linear-gradient(135deg, #22c55e, #4ade80);
    color: black;
    border: none;
    transition: 0.3s;
    cursor: pointer;

}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
}

/* RESULT CARD */
.result-card {
    background: #1e293b;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
    box-shadow: 0px 0px 20px rgba(56,189,248,0.3);
}
.result-text {
    font-size: 34px;
    font-weight: bold;
    color: #22c55e;
}
.conf-text {
    font-size: 22px;
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ----------- HEADER -----------
st.markdown('<div class="title">🚦 Traffic Sign Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Upload any traffic sign & get instant AI prediction</div>', unsafe_allow_html=True)

# ----------- UPLOAD -----------
st.markdown('<div class="upload-box">📤 Upload Any Traffic Sign Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

# ----------- MAIN -----------
if uploaded_file:
    col1, col2 = st.columns([1,1])

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.write("### 🔍 Prediction Panel")

        if st.button("🚀 Predict Traffic Sign"):
            img = np.array(image)
            img = cv2.resize(img, (32, 32))
            img = img / 255.0
            img = img.reshape(1, 32, 32, 3)

            pred = model.predict(img)
            class_id = np.argmax(pred)
            confidence = np.max(pred) * 100

            # RESULT CARD
            st.markdown(f"""
            <div class="result-card">
                <div class="result-text">🚦 {classes[class_id]}</div>
                <div class="conf-text">Confidence: {confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)