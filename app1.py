import streamlit as st
import cv2
import os
import json
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
from pathlib import Path


class TheaterParkingSystem:
    def __init__(self, model_path, total_capacity=100, conf_thres=0.35, min_conf_for_draw=0.5):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.min_conf_for_draw = min_conf_for_draw
        self.total_capacity = total_capacity

        # Get class names
        names_raw = getattr(self.model, "names", None)
        if isinstance(names_raw, dict):
            self.class_names = names_raw
        elif isinstance(names_raw, (list, tuple)):
            self.class_names = {i: n for i, n in enumerate(names_raw)}
        else:
            self.class_names = {}

        # Find vehicle class IDs
        lower_names = {i: str(n).lower() for i, n in self.class_names.items()}
        vehicle_keywords = ['car', 'truck', 'bus', 'motor', 'bike', 'van']
        self.vehicle_classes = [i for i, n in lower_names.items() if any(k in n for k in vehicle_keywords)]

        if len(self.vehicle_classes) == 0:
            self.vehicle_classes = list(self.class_names.keys())

        # Tracking and counting
        self.track_history = defaultdict(list)
        self.last_count_frame = {}
        self.frame_idx = 0
        self.vehicle_counts = {'in': 0, 'out': 0, 'total': 0}
        self.vehicle_type_counts = defaultdict(int)
        self.vehicle_states = {}
        self.counting_line_y = None

        # Color scheme
        self.colors = {
            'bg_secondary': (255, 255, 255),
            'accent_blue': (255, 150, 100),
            'accent_green': (100, 200, 50),
            'accent_red': (100, 100, 255),
            'accent_yellow': (50, 180, 255),
            'text_primary': (40, 40, 50),
            'text_secondary': (80, 80, 90),
            'border': (220, 200, 200),
            'counting_line': (255, 255, 0)
        }

    def get_available_spaces(self):
        occupied = self.vehicle_counts['in'] - self.vehicle_counts['out']
        available = max(0, self.total_capacity - occupied)
        return available, occupied

    def setup_counting_line(self, frame_h, frame_w, ratio=0.5):
        self.counting_line_y = int(frame_h * ratio)

    def draw_counting_line(self, frame):
        h, w = frame.shape[:2]
        cv2.line(frame, (0, self.counting_line_y), (w, self.counting_line_y),
                 self.colors['counting_line'], 3)

        label_x, label_y = 20, self.counting_line_y - 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (label_x, label_y), (label_x + 160, label_y + 28),
                      self.colors['bg_secondary'], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (label_x, label_y), (label_x + 160, label_y + 28),
                      self.colors['counting_line'], 2)
        cv2.putText(frame, "COUNTING LINE", (label_x + 8, label_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_primary'], 1)

    def check_line_crossing(self, track_id, center, cls_id):
        history = self.track_history.get(track_id, [])
        if len(history) < 2:
            return None

        prev_y = history[-2][1]
        curr_y = center[1]

        cooldown_frames = 30
        if track_id in self.last_count_frame and self.frame_idx - self.last_count_frame[track_id] < cooldown_frames:
            return None

        vehicle_type = self.class_names.get(cls_id, 'vehicle').lower()

        if prev_y < self.counting_line_y and curr_y >= self.counting_line_y:
            self.last_count_frame[track_id] = self.frame_idx
            self.vehicle_counts['in'] += 1
            self.vehicle_counts['total'] += 1
            self.vehicle_type_counts[vehicle_type] += 1
            self.vehicle_states[track_id] = {'direction': 'in', 'frame': self.frame_idx}
            return 'in'

        if prev_y > self.counting_line_y and curr_y <= self.counting_line_y:
            self.last_count_frame[track_id] = self.frame_idx
            self.vehicle_counts['out'] += 1
            self.vehicle_counts['total'] += 1
            self.vehicle_states[track_id] = {'direction': 'out', 'frame': self.frame_idx}
            return 'out'

        return None

    def draw_info_panel(self, frame):
        h, w = frame.shape[:2]
        available, occupied = self.get_available_spaces()

        panel_w = 420
        panel_h = 175
        panel_x = 20
        panel_y = 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                      self.colors['bg_secondary'], -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                      self.colors['accent_blue'], 2)

        # Header
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + 30),
                      self.colors['accent_blue'], -1)
        cv2.putText(frame, "VEHICLE IN-OUT MANAGEMENT", (panel_x + 10, panel_y + 21),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

        y_offset = panel_y + 55
        line_height = 28

        counts = [
            (f"IN :  {self.vehicle_counts['in']}", self.colors['accent_green']),
            (f"OUT:  {self.vehicle_counts['out']}", self.colors['accent_red']),
            (f"TOTAL: {self.vehicle_counts['total']}", self.colors['text_primary']),
            (f"AVAILABLE: {available} / {self.total_capacity}", 
             self.colors['accent_green'] if available > 20 else 
             self.colors['accent_yellow'] if available > 5 else self.colors['accent_red']),
        ]

        for i, (text, color) in enumerate(counts):
            cv2.putText(frame, text, (panel_x + 15, y_offset + i * line_height),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    def process_video(self, video_path, progress_callback=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.setup_counting_line(h, w)

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output.name, fourcc, fps, (w, h))

        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_idx += 1

            results = self.model.track(frame, persist=True, conf=self.conf_thres,
                                       tracker="bytetrack.yaml", verbose=False)
            res = results[0]

            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                track_ids = (res.boxes.id.cpu().numpy().astype(int)
                             if res.boxes.id is not None else [None] * len(boxes))

                for idx, box in enumerate(boxes):
                    conf = float(confs[idx])
                    cls_id = int(cls_ids[idx])
                    tid = track_ids[idx]

                    if conf < self.min_conf_for_draw or cls_id not in self.vehicle_classes or tid is None:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    self.track_history[tid].append((cx, cy))
                    if len(self.track_history[tid]) > 30:
                        self.track_history[tid] = self.track_history[tid][-30:]

                    direction = self.check_line_crossing(tid, (cx, cy), cls_id)

                    color = self.colors['accent_green']
                    if abs(cy - self.counting_line_y) <= 30:
                        color = self.colors['accent_yellow']

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"ID:{tid} {self.class_names.get(cls_id, 'vehicle')}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0] + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    if direction:
                        indicator = "↓ IN" if direction == 'in' else "↑ OUT"
                        cv2.putText(frame, indicator, (cx - 20, cy - 20),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

            self.draw_counting_line(frame)
            self.draw_info_panel(frame)

            out.write(frame)
            processed_frames += 1

            if progress_callback and total_frames > 0:
                progress_callback(processed_frames / total_frames)

        cap.release()
        out.release()

        return temp_output.name

    def get_data_summary(self):
        available, occupied = self.get_available_spaces()
        data = {
            'Metric': [
                'Total Vehicles Counted',
                'Vehicles IN',
                'Vehicles OUT',
                'Currently Inside',
                'Available Spaces',
                'Total Capacity'
            ],
            'Count': [
                self.vehicle_counts['total'],
                self.vehicle_counts['in'],
                self.vehicle_counts['out'],
                occupied,
                available,
                self.total_capacity
            ]
        }

        for vehicle_type, count in self.vehicle_type_counts.items():
            data['Metric'].append(f'{vehicle_type.title()} Count')
            data['Count'].append(count)

        return pd.DataFrame(data)


# ─────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Vehicle In-Out Management",
        page_icon="🚗",
        layout="wide"
    )

    # ── Custom CSS ──────────────────────────────
    st.markdown("""
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 0;
        }
        .subtitle {
            color: #6b7280;
            font-size: 1rem;
            margin-top: 0;
        }
        .metric-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            text-align: center;
        }
        .metric-label {
            font-size: 0.8rem;
            color: #6b7280;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1a1a2e;
        }
        .status-ok   { color: #16a34a; }
        .status-warn { color: #d97706; }
        .status-full { color: #dc2626; }
        .section-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1a1a2e;
            border-left: 4px solid #6366f1;
            padding-left: 0.6rem;
            margin: 1.2rem 0 0.6rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────
    st.markdown('<p class="main-title">🚗 Vehicle In-Out Management System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered vehicle tracking · counting · parking availability</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Sidebar ──────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        total_capacity = st.number_input(
            "Parking Capacity",
            min_value=1, max_value=10000,
            value=100, step=1,
            help="Total number of parking spaces available"
        )

        conf_threshold = st.slider(
            "Detection Confidence",
            min_value=0.1, max_value=0.9,
            value=0.35, step=0.05,
            help="Higher = fewer but more accurate detections"
        )

        line_ratio = st.slider(
            "Counting Line Position",
            min_value=0.2, max_value=0.9,
            value=0.5, step=0.05,
            help="Vertical position of the counting line (0 = top, 1 = bottom)"
        )

        st.markdown("---")
        st.header("📋 Info")
        st.markdown("**Model**: YOLO (custom trained)")
        st.markdown("**Tracker**: ByteTrack")
        st.markdown("**Formats**: MP4, AVI, MOV, MKV")
        st.markdown("---")
        st.markdown("**How it works**")
        st.markdown("1. Upload your video")
        st.markdown("2. Set parking capacity")
        st.markdown("3. Click Process Video")
        st.markdown("4. Download results")

    # ── Model path ───────────────────────────────
    model_path = "models/my_trained_model.pt"

    # ── Upload Section ───────────────────────────
    st.markdown('<p class="section-header">📁 Upload Video</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported: MP4, AVI, MOV, MKV"
    )

    if uploaded_file:
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.success(f"✅ **{uploaded_file.name}** — {file_size:.1f} MB loaded")

    # ── Processing ───────────────────────────────
    if uploaded_file is not None:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        # Save uploaded file to temp location
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(uploaded_file.read())
        temp_input.close()

        col1, col2 = st.columns(2)

        with col1:
            process_clicked = st.button("🚀 Process Video", type="primary", use_container_width=True)

        with col2:
            if st.button("🗑️ Clear Results", type="secondary", use_container_width=True):
                if 'processed_video' in st.session_state:
                    try:
                        os.unlink(st.session_state['processed_video'])
                    except Exception:
                        pass
                for key in ['processed_video', 'system', 'processing_done']:
                    st.session_state.pop(key, None)
                st.rerun()

        if process_clicked:
            try:
                with st.spinner("Initializing AI model..."):
                    if not os.path.exists(model_path):
                        st.error(f"❌ Model not found at: `{model_path}`")
                        st.info("Make sure `models/my_trained_model.pt` is in your repo.")
                        st.stop()

                    system = TheaterParkingSystem(
                        model_path=model_path,
                        total_capacity=int(total_capacity),
                        conf_thres=conf_threshold
                    )
                    # Override line ratio from sidebar
                    system._line_ratio = line_ratio
                    st.success("✅ Model loaded!")

                status_placeholder.info("⏳ Processing video — this may take a few minutes...")
                progress_bar = progress_placeholder.progress(0)

                def update_progress(p):
                    progress_bar.progress(min(p, 1.0))
                    status_placeholder.info(f"⏳ Processing: {p * 100:.0f}%")

                # Patch line ratio before processing
                original_setup = system.setup_counting_line
                def patched_setup(frame_h, frame_w, ratio=0.5):
                    original_setup(frame_h, frame_w, ratio=line_ratio)
                system.setup_counting_line = patched_setup

                output_video_path = system.process_video(temp_input.name, update_progress)

                progress_bar.progress(1.0)
                status_placeholder.success("✅ Processing complete!")

                st.session_state['processed_video'] = output_video_path
                st.session_state['system'] = system
                st.session_state['processing_done'] = True

            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                status_placeholder.error("Processing failed.")

    # ── Results ──────────────────────────────────
    if st.session_state.get('processing_done', False):
        system = st.session_state['system']
        available, occupied = system.get_available_spaces()

        st.markdown("---")
        st.markdown('<p class="section-header">📊 Results</p>', unsafe_allow_html=True)

        # Parking availability status banner
        if available == 0:
            st.error(f"🚫 PARKING FULL — 0 spaces left out of {system.total_capacity}")
        elif available <= 5:
            st.warning(f"⚠️ ALMOST FULL — Only {available} spaces left out of {system.total_capacity}")
        elif available <= 20:
            st.info(f"🔵 LIMITED AVAILABILITY — {available} spaces left out of {system.total_capacity}")
        else:
            st.success(f"✅ SPACES AVAILABLE — {available} out of {system.total_capacity} free")

        # Main metrics
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.metric("🔵 Total Counted", system.vehicle_counts['total'])
        with c2:
            st.metric("🟢 Vehicles IN", system.vehicle_counts['in'])
        with c3:
            st.metric("🔴 Vehicles OUT", system.vehicle_counts['out'])
        with c4:
            st.metric("🅿️ Currently Inside", occupied)
        with c5:
            st.metric("✅ Available Spaces", available)

        # Vehicle type breakdown
        if system.vehicle_type_counts:
            st.markdown('<p class="section-header">🚙 Vehicle Types Detected</p>', unsafe_allow_html=True)
            type_cols = st.columns(len(system.vehicle_type_counts))
            for i, (vtype, count) in enumerate(system.vehicle_type_counts.items()):
                with type_cols[i]:
                    st.metric(vtype.title(), count)

        # Downloads
        st.markdown("---")
        st.markdown('<p class="section-header">📥 Downloads</p>', unsafe_allow_html=True)

        dl1, dl2 = st.columns(2)

        with dl1:
            st.markdown("**🎥 Processed Video**")
            video_path = st.session_state.get('processed_video')
            if video_path and os.path.exists(video_path):
                with open(video_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=f.read(),
                        file_name=f"vehicle_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4",
                        type="primary",
                        use_container_width=True
                    )

        with dl2:
            st.markdown("**📊 CSV Report**")
            csv_df = system.get_data_summary()
            st.download_button(
                label="⬇️ Download CSV Report",
                data=csv_df.to_csv(index=False),
                file_name=f"vehicle_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="secondary",
                use_container_width=True
            )

        # Preview table
        with st.expander("📋 View Full Report"):
            st.dataframe(csv_df, use_container_width=True)

        # Cleanup temp input
        try:
            if 'temp_input' in dir() and os.path.exists(temp_input.name):
                os.unlink(temp_input.name)
        except Exception:
            pass


if __name__ == "__main__":
    main()