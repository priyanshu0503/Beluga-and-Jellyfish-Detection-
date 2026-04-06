import streamlit as st
from ultralytics import YOLO
import pandas as pd
import tempfile
import os
from collections import defaultdict

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Marine Detection",
    layout="wide"
)

# -------------------------------
# CUSTOM DESIGN (CSS)
# -------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e2e8f0;
}

/* Title */
h1 {
    font-size: 3rem !important;
    font-weight: 600;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Cards */
.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

/* Upload */
[data-testid="stFileUploader"] {
    border: 2px dashed #38bdf8;
    border-radius: 12px;
    padding: 20px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #22c55e);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}

header {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("Settings")

st.sidebar.markdown("### Detection Settings")

model_choice = st.sidebar.selectbox(
    "Model",
    ["Beluga", "Jellyfish"]
)

confidence = st.sidebar.slider(
    "Confidence",
    0.0, 1.0, 0.25,
    help="Minimum confidence required to display a detection"
)

iou = st.sidebar.slider(
    "IoU",
    0.0, 1.0, 0.45,
    help="Controls overlap filtering between bounding boxes"
)


# -------------------------------
# LOAD MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_model(choice):
    if choice == "Beluga":
        return YOLO("models/beluga/best.pt")
    else:
        return YOLO("models/jellyfish/best.pt")

model = load_model(model_choice)

# -------------------------------
# HERO SECTION
# -------------------------------
st.markdown("""
<div style="text-align:center; padding: 20px;">
    <h1>Marine Detection System </h1>
    <p style="color:#94a3b8; font-size:18px;">
        Detect Beluga Whales and Jellyfish using YOLO models.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# PROCESSING
# -------------------------------
all_data = []
overall_counts = defaultdict(int)

if uploaded_files:

    progress_bar = st.progress(0)
    total_files = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):

        file_extension = os.path.splitext(uploaded_file.name)[1]

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp.write(uploaded_file.read())
        temp.close()

        results = model(temp.name, conf=confidence, iou=iou)

        class_counts = defaultdict(int)

        for r in results:
            if r.boxes is None:
                continue

            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i])
                class_name = model.names[cls_id]
                class_counts[class_name] += 1
                overall_counts[class_name] += 1

        for r in results:
            if r.boxes is None:
                continue

            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i])
                class_name = model.names[cls_id]

                confidence_score = float(r.boxes.conf[i])

                x_center = float(r.boxes.xywh[i][0])
                y_center = float(r.boxes.xywh[i][1])
                width = float(r.boxes.xywh[i][2])
                height = float(r.boxes.xywh[i][3])

                all_data.append({
                    "image_name": uploaded_file.name,
                    "class": class_name,
                    "confidence": round(confidence_score, 3),
                    "x_center": round(x_center, 2),
                    "y_center": round(y_center, 2),
                    "width": round(width, 2),
                    "height": round(height, 2),
                    "count_per_class": class_counts[class_name]
                })

        # -------------------------------
        # DISPLAY
        # -------------------------------
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(results[0].plot(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detections")

            if class_counts:
                for cls, count in class_counts.items():
                    st.metric(cls, count)
            else:
                st.write("No objects detected")

            st.markdown('</div>', unsafe_allow_html=True)

        os.remove(temp.name)

        progress_bar.progress((idx + 1) / total_files)

    # -------------------------------
    # DATA TABLE
    # -------------------------------
    df = pd.DataFrame(all_data)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Detection Analytics")
    st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------
    # OVERALL SUMMARY
    # -------------------------------
    st.subheader("Overall Summary")

    if overall_counts:
        cols = st.columns(len(overall_counts))
        for i, (cls, count) in enumerate(overall_counts.items()):
            cols[i].metric(cls, count)
    else:
        st.write("No detections found")

    # -------------------------------
    # DOWNLOAD
    # -------------------------------
    csv = df.to_csv(index=False)

    st.download_button(
        "Download CSV",
        csv,
        "marine_detection_results.csv",
        use_container_width=True
    )