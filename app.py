"""
EcoVision AI - Real-Time AI Waste Classification Dashboard
Author: College AI Mini Project Team
Framework: Streamlit & Ultralytics YOLO11n

Pages:
  1. 🌿 Home Dashboard
  2. 📹 Live Webcam Detection
  3. 🖼️ Upload Image
  4. 📊 Detection History & Analytics
  5. ♻️ Waste Disposal Guide
  6. ℹ️ About Project
"""

import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import streamlit as st
import matplotlib.pyplot as plt

# Add root directory to python path for utils
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from utils.disposal_guide import (
    WASTE_CATEGORIES,
    NAME_TO_ID,
    get_category_info,
    get_all_categories,
    get_disclaimer,
)
from utils.detector import WasteDetector

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoVision AI - Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS Stylesheet
css_path = BASE_DIR / "assets" / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

MODEL_PATH = BASE_DIR / "model" / "waste_yolo11n.pt"

# Cache detector so we do not reload the model on every UI interaction
@st.cache_resource(show_spinner=False)
def load_cached_detector(model_file: str, conf_threshold: float = 0.40):
    try:
        return WasteDetector(model_path=model_file, conf_threshold=conf_threshold)
    except Exception as e:
        return None


# -----------------------------------------------------------------------------
# Navigation Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌿 EcoVision AI")
    st.caption("AI-Powered Waste Segregation System")
    st.markdown("---")

    page = st.radio(
        "Navigation Menu",
        [
            "🌿 Home Dashboard",
            "📹 Live Webcam Detection",
            "🖼️ Upload Image",
            "📊 Detection History",
            "♻️ Waste Disposal Guide",
            "ℹ️ About Project"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    if MODEL_PATH.exists():
        st.success("✅ Model: `waste_yolo11n.pt` Ready")
    else:
        st.warning("⚠️ Model: Weights not found in `model/`")

    st.markdown("---")
    st.caption("College AI Mini Project • Windows 11 CPU")


# -----------------------------------------------------------------------------
# PAGE 1: Home Dashboard
# -----------------------------------------------------------------------------
if page == "🌿 Home Dashboard":
    st.markdown(
        """
        <div class="eco-header">
            <h1>🌿 EcoVision AI: Smart Waste Classification</h1>
            <p>Real-time computer vision system classifying waste into 5 essential segregation categories.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Overview Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="eco-card">
                <h3>🎯 Mission</h3>
                <p>Automate household and campus waste sorting using real-time YOLO11n object detection to foster sustainable recycling habits.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="eco-card">
                <h3>⚡ Architecture</h3>
                <p>Engineered for standard laptop hardware running CPU inference on Intel multi-core architecture without dedicated GPU requirements.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="eco-card">
                <h3>♻️ Impact</h3>
                <p>Diverts organic material from methane-generating landfills and ensures high-purity segregation for paper, plastic, glass, and metal.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🏷️ Supported Waste Categories")
    cat_cols = st.columns(5)
    for idx, (cat_id, info) in enumerate(WASTE_CATEGORIES.items()):
        with cat_cols[idx]:
            st.markdown(
                f"""
                <div class="eco-card" style="border-top: 4px solid {info['color_hex']}; text-align: center;">
                    <div style="font-size: 2.2rem; margin-bottom: 0.3rem;">{info['icon']}</div>
                    <h4 style="margin: 0; color: {info['color_hex']};">{info['name']}</h4>
                    <p style="font-size: 0.8rem; color: #6B7280; margin-top: 0.4rem;">{info['bin_type']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("### 🚀 Quick Navigation")
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        st.info("📹 **Live Webcam**: Point your laptop webcam at waste objects for instant real-time bounding boxes and segregation advice.")
    with q_col2:
        st.success("🖼️ **Upload Image**: Select waste photos (JPG, PNG) from your laptop to test detection accuracy and explore disposal tips.")

    st.markdown(f"<div class='eco-disclaimer'>{get_disclaimer()}</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 2: Live Webcam Detection
# -----------------------------------------------------------------------------
elif page == "📹 Live Webcam Detection":
    st.markdown("## 📹 Live Webcam Waste Detection")
    st.markdown("Point your laptop webcam at waste items (bottles, paper, cans, peelings) to see real-time bounding boxes and disposal suggestions.")

    # Check for model weights
    if not MODEL_PATH.exists():
        st.error(
            "❌ Model weights not found at `model/waste_yolo11n.pt`.\n\n"
            "Please train the model first by executing `python train.py` in your terminal."
        )
        st.stop()

    # Settings Row
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 2])
    with ctrl_col1:
        conf_slider = st.slider("Detection Confidence Threshold", min_value=0.20, max_value=0.90, value=0.40, step=0.05)
    with ctrl_col2:
        start_btn = st.button("▶️ Start Camera", use_container_width=True, type="primary")
    with ctrl_col3:
        stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)

    if start_btn:
        st.session_state.camera_running = True
    if stop_btn:
        st.session_state.camera_running = False

    # Video stream placeholder
    stream_placeholder = st.empty()
    status_placeholder = st.empty()
    results_placeholder = st.empty()

    if st.session_state.camera_running:
        status_placeholder.info("Connecting to built-in webcam via DirectShow...")
        detector = load_cached_detector(str(MODEL_PATH), conf_threshold=conf_slider)
        if detector is None:
            st.error("Failed to initialize detection engine.")
            st.session_state.camera_running = False
            st.stop()

        # Connect to webcam using DirectShow
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # Fallback to default backend
            cap.release()
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            status_placeholder.error(
                "❌ Could not access your webcam. Please check:\n"
                "1. Windows Camera Privacy Settings (Settings -> Privacy & Security -> Camera).\n"
                "2. Ensure no other application (Zoom, Teams, Chrome) is using the camera.\n"
                "3. Check your HP keyboard camera privacy key."
            )
            st.session_state.camera_running = False
        else:
            status_placeholder.success("🟢 Camera streaming active. Point waste objects towards camera.")
            fps_tracker = []

            try:
                while st.session_state.camera_running:
                    t_start = time.perf_counter()
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        status_placeholder.warning("⚠️ Failed to grab video frame.")
                        break

                    # Run YOLO detection
                    annotated_frame, detections = detector.predict(frame, draw_hud=True)

                    # Calculate FPS
                    t_latency = (time.perf_counter() - t_start) * 1000.0
                    fps_val = 1000.0 / t_latency if t_latency > 0 else 0
                    fps_tracker.append(fps_val)
                    if len(fps_tracker) > 10:
                        fps_tracker.pop(0)
                    avg_fps = np.mean(fps_tracker)

                    # Overlay FPS HUD
                    cv2.putText(
                        annotated_frame,
                        f"FPS: {avg_fps:.1f} ({t_latency:.0f}ms)",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    # Convert BGR to RGB for Streamlit display
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    stream_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    # Log detections into history and display info cards
                    if detections:
                        cards_html = "<div style='display: flex; gap: 10px; flex-wrap: wrap; margin-top: 15px;'>"
                        for det in detections:
                            info = det["info"]
                            cname = det["class_name"]
                            c_conf = det["confidence"] * 100

                            # Add to session history
                            st.session_state.history.append({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "category": cname,
                                "confidence": f"{c_conf:.1f}%",
                                "source": "Webcam"
                            })

                            cards_html += f"""
                            <div class="eco-card" style="flex: 1; min-width: 200px; border-left: 5px solid {info['color_hex']}; padding: 10px;">
                                <strong>{info['icon']} {cname} ({c_conf:.1f}%)</strong><br/>
                                <small style="color: {info['color_hex']};"><strong>{info['bin_type']}</strong></small><br/>
                                <small>{info['disposal_suggestion']}</small>
                            </div>
                            """
                        cards_html += "</div>"
                        results_placeholder.markdown(cards_html, unsafe_allow_html=True)
                    else:
                        results_placeholder.caption("No waste object currently detected in view.")

            finally:
                cap.release()
                status_placeholder.info("Camera stopped and hardware released.")
    else:
        stream_placeholder.info("Click '▶️ Start Camera' above to activate live detection.")

    st.markdown("---")
    st.markdown("### 📸 Alternate: Snapshot Input")
    st.caption("You can also take a quick snapshot using your camera:")
    cam_snap = st.camera_input("Capture Waste Snapshot")
    if cam_snap is not None:
        pil_snap = Image.open(cam_snap)
        detector = load_cached_detector(str(MODEL_PATH), conf_threshold=conf_slider)
        if detector:
            ann_bgr, dets = detector.predict(pil_snap, draw_hud=True)
            ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, caption="Snapshot Detection Result", use_container_width=True)

            if dets:
                st.success(f"Detected {len(dets)} waste item(s):")
                for d in dets:
                    st.write(f"- **{d['info']['icon']} {d['class_name']}** ({d['confidence']*100:.1f}%) -> {d['info']['disposal_suggestion']}")
            else:
                st.warning("No waste object detected in snapshot.")


# -----------------------------------------------------------------------------
# PAGE 3: Upload Image Detection
# -----------------------------------------------------------------------------
elif page == "🖼️ Upload Image":
    st.markdown("## 🖼️ Upload Waste Image for Classification")
    st.markdown("Upload a photo containing one or more waste objects (`.jpg`, `.jpeg`, `.png`) to run YOLO11n detection.")

    if not MODEL_PATH.exists():
        st.error(
            "❌ Model weights not found at `model/waste_yolo11n.pt`.\n\n"
            "Please train the model first by executing `python train.py` in your terminal."
        )
        st.stop()

    conf_slider = st.slider("Detection Confidence Threshold", min_value=0.20, max_value=0.90, value=0.40, step=0.05)
    uploaded_file = st.file_uploader("Choose a waste image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read uploaded image: {e}")
            st.stop()

        detector = load_cached_detector(str(MODEL_PATH), conf_threshold=conf_slider)
        if detector is None:
            st.error("Failed to initialize detector.")
            st.stop()

        col_orig, col_det = st.columns(2)
        with col_orig:
            st.markdown("#### 📷 Original Image")
            st.image(image, use_container_width=True)

        # Run inference
        t0 = time.perf_counter()
        annotated_bgr, detections = detector.predict(image, draw_hud=True)
        latency = (time.perf_counter() - t0) * 1000.0
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        with col_det:
            st.markdown("#### 🔍 Detection Result")
            st.image(annotated_rgb, use_container_width=True)
            st.caption(f"⚡ CPU Latency: {latency:.1f} ms")

        st.markdown("---")
        st.markdown("### 📋 Detected Waste Items & Segregation Advice")

        if not detections:
            st.warning("No waste items detected with the current confidence threshold. Try lowering the threshold slider above.")
        else:
            for idx, det in enumerate(detections, start=1):
                info = det["info"]
                cname = det["class_name"]
                conf_pct = det["confidence"] * 100

                # Record in session history
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "category": cname,
                    "confidence": f"{conf_pct:.1f}%",
                    "source": "Upload"
                })

                with st.container():
                    st.markdown(
                        f"""
                        <div class="eco-card" style="border-left: 6px solid {info['color_hex']};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h3 style="margin: 0; color: {info['color_hex']};">{info['icon']} Object #{idx}: {cname}</h3>
                                <span style="font-size: 1.1rem; font-weight: 700; color: {info['color_hex']};">{conf_pct:.1f}% Confidence</span>
                            </div>
                            <p style="margin-top: 0.5rem; font-weight: 600;">Target Receptacle: {info['bin_type']}</p>
                            <p><strong>Disposal Instruction:</strong> {info['disposal_suggestion']}</p>
                            <p style="font-size: 0.85rem; color: #4B5563;"><strong>Eco Tip:</strong> {info['handling_tips'][0]}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.markdown(f"<div class='eco-disclaimer'>{get_disclaimer()}</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 4: Detection History & Statistics
# -----------------------------------------------------------------------------
elif page == "📊 Detection History":
    st.markdown("## 📊 Session Detection History & Analytics")
    st.markdown("Interactive analytics tracking all waste items identified during your current session.")

    history = st.session_state.history

    if not history:
        st.info("ℹ️ No detections recorded in this session yet. Test the Webcam or Upload an image to view live analytics.")
    else:
        # KPI metrics
        total_detections = len(history)
        cat_counts = {name: 0 for name in ["Organic", "Plastic", "Glass", "Paper", "Metal"]}
        for item in history:
            cat = item["category"]
            if cat in cat_counts:
                cat_counts[cat] += 1

        kpi_cols = st.columns(6)
        with kpi_cols[0]:
            st.metric("Total Scanned", total_detections)
        with kpi_cols[1]:
            st.metric("🌱 Organic", cat_counts["Organic"])
        with kpi_cols[2]:
            st.metric("🧴 Plastic", cat_counts["Plastic"])
        with kpi_cols[3]:
            st.metric("🍾 Glass", cat_counts["Glass"])
        with kpi_cols[4]:
            st.metric("📄 Paper", cat_counts["Paper"])
        with kpi_cols[5]:
            st.metric("🥫 Metal", cat_counts["Metal"])

        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)

        # Bar Chart
        with chart_col1:
            st.markdown("#### Category Distribution (Bar Chart)")
            fig, ax = plt.subplots(figsize=(6, 4))
            cats = list(cat_counts.keys())
            vals = list(cat_counts.values())
            colors = ["#2E7D32", "#0288D1", "#00897B", "#F57C00", "#78909C"]
            bars = ax.bar(cats, vals, color=colors, edgecolor="black", linewidth=0.8)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title("Detections per Category", fontsize=12, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
            st.pyplot(fig)

        # Pie Chart
        with chart_col2:
            st.markdown("#### Segregation Proportions (Pie Chart)")
            nonzero_cats = [c for c, v in cat_counts.items() if v > 0]
            nonzero_vals = [v for v in cat_counts.values() if v > 0]
            nonzero_colors = [colors[cats.index(c)] for c in nonzero_cats]

            if nonzero_vals:
                fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
                ax_pie.pie(
                    nonzero_vals,
                    labels=nonzero_cats,
                    colors=nonzero_colors,
                    autopct="%1.1f%%",
                    startangle=140,
                    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
                )
                ax_pie.set_title("Waste Stream Composition", fontsize=12, fontweight="bold")
                st.pyplot(fig_pie)
            else:
                st.write("No positive counts to render pie chart.")

        st.markdown("---")
        st.markdown("#### 📜 Timestamped Event Log")
        st.dataframe(history, use_container_width=True)

        if st.button("🗑️ Clear History Log"):
            st.session_state.history = []
            st.rerun()


# -----------------------------------------------------------------------------
# PAGE 5: Waste Disposal Guide
# -----------------------------------------------------------------------------
elif page == "♻️ Waste Disposal Guide":
    st.markdown("## ♻️ Comprehensive Waste Segregation & Disposal Guide")
    st.markdown("Educational handbook explaining proper segregation techniques and recycling principles.")

    for cat_id, info in WASTE_CATEGORIES.items():
        with st.expander(f"{info['icon']} **{info['name']} Waste Guide** — {info['bin_type']}", expanded=True):
            col_l, col_r = st.columns([3, 2])
            with col_l:
                st.markdown(f"### Disposal Recommendation")
                st.write(info["disposal_suggestion"])
                st.markdown("#### Best Practices & Handling Tips")
                for tip in info["handling_tips"]:
                    st.write(f"- {tip}")
            with col_r:
                st.markdown("#### Environmental Impact")
                st.info(info["environmental_impact"])

    st.markdown(f"<div class='eco-disclaimer'>{get_disclaimer()}</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 6: About Project
# -----------------------------------------------------------------------------
elif page == "ℹ️ About Project":
    st.markdown("## ℹ️ About EcoVision AI")
    st.markdown(
        """
        **EcoVision AI** is an AI-powered solid waste classification and segregation assistant developed as a 
        **Second-Year Computer Science & Engineering College AI Mini Project**.
        """
    )

    st.markdown("### 🛠️ Technology Stack")
    st.markdown(
        """
        - **Computer Vision & Object Detection**: Ultralytics YOLO11n (Nano architecture, ~2.6M parameters)
        - **Inference Runtime**: PyTorch (CPU-optimized build, zero external GPU dependency)
        - **Streamlit Web Dashboard**: Interactive UI with session analytics and HUD overlays
        - **Frame Processing**: OpenCV (DirectShow Windows 11 backend)
        - **Programming Language**: Python 3.11.9 (64-bit)
        """
    )

    st.markdown("---")
    st.markdown("### 📊 Dataset Attribution & Academic Integrity (CC BY 4.0)")
    st.markdown(
        """
        - **Dataset Source**: *Garbage Classification 3 (Multi-Class Waste Detection)* via Roboflow Universe
        - **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
        - **Modifications**: 
          - Merged `BIODEGRADABLE` into **Organic**.
          - Merged `PAPER` and `CARDBOARD` into **Paper**.
          - Retained **Plastic**, **Glass**, and **Metal** as discrete classes.
          - Excluded `CLOTH` annotations to strictly optimize for 5 core solid waste streams.
          - Formally verified zero data leakage between train, validation, and test splits using SHA-256 image hashes.
        """
    )

    # Display evaluation plots if available
    cm_path = BASE_DIR / "model" / "confusion_matrix.png"
    res_path = BASE_DIR / "model" / "results.png"

    if cm_path.exists() or res_path.exists():
        st.markdown("---")
        st.markdown("### 📈 Model Evaluation Artifacts")
        plot_col1, plot_col2 = st.columns(2)
        if cm_path.exists():
            with plot_col1:
                st.markdown("#### Authentic Confusion Matrix")
                st.image(str(cm_path), caption="Validation Confusion Matrix", use_container_width=True)
        if res_path.exists():
            with plot_col2:
                st.markdown("#### Training & Validation Loss Curves")
                st.image(str(res_path), caption="Training Results (mAP, Loss)", use_container_width=True)

    st.markdown("---")
    st.caption("EcoVision AI Project • Built for College Mini Project Evaluation")
