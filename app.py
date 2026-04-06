import streamlit as st
from ultralytics import YOLO
import pandas as pd
import tempfile
import os
from collections import defaultdict

st.title("Marine Detection System")

# -------------------------------
# MODEL SELECTION
# -------------------------------
model_choice = st.selectbox(
    "Select Model",
    ["Beluga", "Jellyfish"]
)

if model_choice == "Beluga":
    model = YOLO("beluga_best_model.pt")
else:
    model = YOLO("jellyfish_best_model.pt")

# -------------------------------
# FILE UPLOAD (MULTIPLE)
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

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

        # -------------------------------
        # RUN DETECTION
        # -------------------------------
        results = model(temp.name, conf=0.2, iou=0.4, agnostic_nms=False)

        class_counts = defaultdict(int)

        # FIRST PASS → count per class
        for r in results:
            if r.boxes is None:
                continue

            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i])
                class_name = model.names[cls_id]
                class_counts[class_name] += 1
                overall_counts[class_name] += 1

        # SECOND PASS → store detection data
        for r in results:
            if r.boxes is None:
                continue

            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i])
                class_name = model.names[cls_id]

                confidence = float(r.boxes.conf[i])

                x_center = float(r.boxes.xywh[i][0])
                y_center = float(r.boxes.xywh[i][1])
                width = float(r.boxes.xywh[i][2])
                height = float(r.boxes.xywh[i][3])

                all_data.append({
                    "image_name": uploaded_file.name,
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "x_center": round(x_center, 2),
                    "y_center": round(y_center, 2),
                    "width": round(width, 2),
                    "height": round(height, 2),
                    "count_per_class": class_counts[class_name]
                })

        # -------------------------------
        # DISPLAY RESULT PER IMAGE
        # -------------------------------
        st.subheader(f"Result: {uploaded_file.name}")
        st.image(results[0].plot(), caption="Detection Result")

        st.write("Summary:")
        if class_counts:
            for cls, count in class_counts.items():
                st.write(f"{cls}: {count}")
        else:
            st.write("No objects detected")

        # cleanup temp file
        os.remove(temp.name)

        # update progress bar
        progress_bar.progress((idx + 1) / total_files)

    # -------------------------------
    # FINAL COMBINED TABLE
    # -------------------------------
    df = pd.DataFrame(all_data)

    st.subheader("Combined Detection Table")
    st.dataframe(df)

    # -------------------------------
    # OVERALL SUMMARY
    # -------------------------------
    st.subheader("Overall Summary (All Images)")

    if overall_counts:
        for cls, count in overall_counts.items():
            st.write(f"{cls}: {count}")
    else:
        st.write("No objects detected in dataset")

    # -------------------------------
    # DOWNLOAD CSV
    # -------------------------------
    csv = df.to_csv(index=False)
    st.download_button("Download Full CSV", csv, "all_results.csv")
