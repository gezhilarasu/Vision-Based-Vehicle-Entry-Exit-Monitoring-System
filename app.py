import streamlit as st
import cv2
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
import tempfile
from pathlib import Path

class VehicleManagementSystem:
    def __init__(self, model_path, conf_thres=0.35, min_conf_for_draw=0.5):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.min_conf_for_draw = min_conf_for_draw
        
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
            'bg_primary': (245, 245, 250),
            'bg_secondary': (255, 255, 255),
            'accent_blue': (100, 150, 255),
            'accent_green': (50, 200, 100),
            'accent_red': (255, 100, 100),
            'text_primary': (40, 40, 50),
            'text_secondary': (80, 80, 90),
            'counting_line': (0, 255, 255)
        }

    def setup_counting_line(self, frame_h, frame_w, ratio=0.7):
        """Setup counting line position"""
        self.counting_line_y = int(frame_h * ratio)

    def draw_counting_line(self, frame):
        """Draw the vehicle counting line"""
        h, w = frame.shape[:2]
        
        # Draw counting line
        cv2.line(frame, (0, self.counting_line_y), (w, self.counting_line_y), 
                self.colors['counting_line'], 3)
        
        # Label
        cv2.rectangle(frame, (20, self.counting_line_y - 30), (180, self.counting_line_y - 5), 
                     self.colors['bg_secondary'], -1)
        cv2.rectangle(frame, (20, self.counting_line_y - 30), (180, self.counting_line_y - 5), 
                     self.colors['counting_line'], 2)
        cv2.putText(frame, "COUNTING LINE", (25, self.counting_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_primary'], 1)

    def check_line_crossing(self, track_id, center, cls_id):
        """Check if vehicle crossed the counting line"""
        history = self.track_history.get(track_id, [])
        if len(history) < 2:
            return None

        prev_y = history[-2][1]
        curr_y = center[1]

        cooldown_frames = 30
        if track_id in self.last_count_frame and self.frame_idx - self.last_count_frame[track_id] < cooldown_frames:
            return None

        vehicle_type = self.class_names.get(cls_id, 'vehicle').lower()

        # Moving down (entering)
        if prev_y < self.counting_line_y and curr_y >= self.counting_line_y:
            self.last_count_frame[track_id] = self.frame_idx
            self.vehicle_counts['in'] += 1
            self.vehicle_counts['total'] += 1
            self.vehicle_type_counts[vehicle_type] += 1
            self.vehicle_states[track_id] = {'direction': 'in', 'frame': self.frame_idx}
            return 'in'

        # Moving up (exiting)
        if prev_y > self.counting_line_y and curr_y <= self.counting_line_y:
            self.last_count_frame[track_id] = self.frame_idx
            self.vehicle_counts['out'] += 1
            self.vehicle_counts['total'] += 1
            self.vehicle_states[track_id] = {'direction': 'out', 'frame': self.frame_idx}
            return 'out'

        return None

    def draw_info_panel(self, frame):
        """Draw compact info panel with counts"""
        h, w = frame.shape[:2]
        
        # Main panel
        panel_w = 400
        panel_h = 140
        panel_x = 20
        panel_y = 20
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     self.colors['bg_secondary'], -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     self.colors['accent_blue'], 2)
        
        # Header
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + 30), 
                     self.colors['accent_blue'], -1)
        cv2.putText(frame, "VEHICLE IN-OUT MANAGEMENT", (panel_x + 10, panel_y + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        
        # Counts
        y_offset = panel_y + 55
        line_height = 25
        
        counts = [
            f"IN: {self.vehicle_counts['in']}",
            f"OUT: {self.vehicle_counts['out']}",
            f"TOTAL: {self.vehicle_counts['total']}"
        ]
        
        for i, count in enumerate(counts):
            cv2.putText(frame, count, (panel_x + 15, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text_primary'], 2)

    def process_video(self, video_path, progress_callback=None):
        """Process video and return processed frames"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.setup_counting_line(h, w)
        
        # Initialize video writer for output
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output.name, fourcc, fps, (w, h))
        
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_idx += 1
            
            # Run YOLO detection and tracking
            results = self.model.track(frame, persist=True, conf=self.conf_thres,
                                     tracker="bytetrack.yaml", verbose=False)
            res = results[0]
            
            # Process detections
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                track_ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [None] * len(boxes)
                
                for idx, box in enumerate(boxes):
                    conf = float(confs[idx])
                    cls_id = int(cls_ids[idx])
                    tid = track_ids[idx]
                    
                    if conf < self.min_conf_for_draw or cls_id not in self.vehicle_classes or tid is None:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
                    # Update tracking history
                    self.track_history[tid].append((cx, cy))
                    if len(self.track_history[tid]) > 30:
                        self.track_history[tid] = self.track_history[tid][-30:]
                    
                    # Check for line crossing
                    direction = self.check_line_crossing(tid, (cx, cy), cls_id)
                    
                    # Draw bounding box
                    color = self.colors['accent_green']
                    if abs(cy - self.counting_line_y) <= 30:
                        color = self.colors['accent_red']
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label
                    label = f"ID:{tid} {self.class_names.get(cls_id, 'vehicle')}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0] + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Direction indicator
                    if direction:
                        indicator = "↓ IN" if direction == 'in' else "↑ OUT"
                        cv2.putText(frame, indicator, (cx - 20, cy - 20), 
                                  cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)
            
            # Draw UI elements
            self.draw_counting_line(frame)
            self.draw_info_panel(frame)
            
            # Write frame
            out.write(frame)
            processed_frames += 1
            
            # Update progress
            if progress_callback:
                progress = processed_frames / total_frames
                progress_callback(progress)
        
        cap.release()
        out.release()
        
        return temp_output.name

    def get_data_summary(self):
        """Get summary data for CSV export"""
        data = {
            'Metric': ['Total Vehicles', 'Vehicles In', 'Vehicles Out'],
            'Count': [self.vehicle_counts['total'], self.vehicle_counts['in'], self.vehicle_counts['out']]
        }
        
        # Add vehicle type counts
        for vehicle_type, count in self.vehicle_type_counts.items():
            data['Metric'].append(f'{vehicle_type.title()} Count')
            data['Count'].append(count)
        
        return pd.DataFrame(data)


def main():
    st.set_page_config(
        page_title="Vehicle In-Out Management",
        page_icon="🚗",
        layout="wide"
    )
    
    # Simple Header
    st.title("🚗 Vehicle In-Out Management System")
    st.markdown("AI-powered vehicle tracking and counting system")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 System Info")
        st.markdown("**Model**: YOLO Detection")
        st.markdown("**Formats**: MP4, AVI, MOV, MKV")
        
        st.markdown("---")
        st.header("🔧 How It Works")
        st.markdown("1. Upload video file")
        st.markdown("2. AI processes frames")
        st.markdown("3. Download results")
    
    # Fixed model path and confidence
    model_path = "models/my_trained_model.pt"
    conf_threshold = 0.35
    
    # Video Upload Section
    st.header("📁 Upload Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file:
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.success(f"✅ File loaded: **{uploaded_file.name}** ({file_size:.1f} MB)")
    
    # Processing Status
    if uploaded_file is not None:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(uploaded_file.read())
        temp_input.close()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Process Video", type="primary", use_container_width=True):
                try:
                    # Initialize system
                    with st.spinner("Initializing AI system..."):
                        if not os.path.exists(model_path):
                            st.error(f"❌ Model file not found at: {model_path}")
                            st.stop()
                        
                        system = VehicleManagementSystem(model_path, conf_threshold)
                        st.success("✅ System initialized!")
                    
                    # Process video
                    status_placeholder.info("Processing video...")
                    progress_bar = progress_placeholder.progress(0)
                    
                    def update_progress(progress):
                        progress_bar.progress(progress)
                        status_placeholder.info(f"Processing: {progress*100:.0f}%")
                    
                    output_video_path = system.process_video(temp_input.name, update_progress)
                    
                    status_placeholder.success("✅ Processing completed!")
                    progress_bar.progress(1.0)
                    
                    # Store results
                    st.session_state['processed_video'] = output_video_path
                    st.session_state['system'] = system
                    st.session_state['processing_done'] = True
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    status_placeholder.error("❌ Processing failed!")
        
        with col2:
            if st.button("🗑️ Clear Results", type="secondary", use_container_width=True):
                if 'processed_video' in st.session_state:
                    try:
                        os.unlink(st.session_state['processed_video'])
                    except:
                        pass
                for key in ['processed_video', 'system', 'processing_done']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Results Section
    if st.session_state.get('processing_done', False):
        st.markdown("---")
        st.header("📈 Results")
        
        system = st.session_state['system']
        
        # Main metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔵 Total Vehicles", system.vehicle_counts['total'])
        
        with col2:
            st.metric("🟢 Vehicles In", system.vehicle_counts['in'])
        
        with col3:
            st.metric("🔴 Vehicles Out", system.vehicle_counts['out'])
        
        # Vehicle types
        if system.vehicle_type_counts:
            st.subheader("🚙 Vehicle Types")
            type_cols = st.columns(len(system.vehicle_type_counts))
            for i, (vehicle_type, count) in enumerate(system.vehicle_type_counts.items()):
                with type_cols[i]:
                    st.metric(vehicle_type.title(), count)
        
        # Downloads
        st.markdown("---")
        st.subheader("📥 Downloads")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎥 Processed Video**")
            if os.path.exists(st.session_state['processed_video']):
                with open(st.session_state['processed_video'], 'rb') as f:
                    st.download_button(
                        label="Download Video",
                        data=f.read(),
                        file_name=f"processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4",
                        type="primary"
                    )
        
        with col2:
            st.markdown("**📊 CSV Report**")
            csv_data = system.get_data_summary()
            csv_string = csv_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_string,
                file_name=f"vehicle_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="secondary"
            )
        
        # Clean up temporary input file
        try:
            os.unlink(temp_input.name)
        except:
            pass


if __name__ == "__main__":
    main()