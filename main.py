import cv2
import os
import json
import time
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
import threading

class TheaterParkingSystem:
    def __init__(self, model_path, total_capacity=100, conf_thres=0.35, min_conf_for_draw=0.5):
        print(f"🎭 Initializing Theater Parking System...")
        print(f"Loading model from: {model_path}")
        
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
            print("⚠ No vehicle classes detected — using all classes.")
            self.vehicle_classes = list(self.class_names.keys())

        print(f"Detected vehicle class ids: {self.vehicle_classes}")

        # Tracking and counting
        self.track_history = defaultdict(list)
        self.last_count_frame = {}
        self.frame_idx = 0
        self.vehicle_counts = {'in': 0, 'out': 0}
        self.vehicle_states = {}
        self.counting_line_y = None
        
        # Parking data file for persistence
        self.data_file = "parking_data.json"
        self.load_parking_data()
        
        # Display settings
        self.display_fullscreen = False
        
        print(f"🚗 System initialized - Total Capacity: {self.total_capacity}")
        print(f"📊 Current Status - IN: {self.vehicle_counts['in']}, OUT: {self.vehicle_counts['out']}")

    def load_parking_data(self):
        """Load existing parking data from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.vehicle_counts = data.get('counts', {'in': 0, 'out': 0})
                    print(f"📁 Loaded previous data: IN={self.vehicle_counts['in']}, OUT={self.vehicle_counts['out']}")
            except Exception as e:
                print(f"⚠ Error loading data: {e}")

    def save_parking_data(self):
        """Save current parking data to file"""
        try:
            data = {
                'counts': self.vehicle_counts,
                'last_updated': datetime.now().isoformat(),
                'capacity': self.total_capacity
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠ Error saving data: {e}")

    def get_available_spaces(self):
        """Calculate available parking spaces"""
        occupied = self.vehicle_counts['in'] - self.vehicle_counts['out']
        available = max(0, self.total_capacity - occupied)
        return available, occupied

    def setup_counting_line(self, frame_h, frame_w, ratio=0.5):
        """Setup counting line position"""
        self.counting_line_y = int(frame_h * ratio)
        print(f"📏 Counting line positioned at Y={self.counting_line_y}")

    def draw_counting_line(self, frame):
        """Draw the vehicle counting line"""
        h, w = frame.shape[:2]
        cv2.line(frame, (0, self.counting_line_y), (w, self.counting_line_y), (0, 255, 255), 4)
        cv2.putText(frame, "COUNTING LINE", (10, self.counting_line_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    def check_line_crossing(self, track_id, center):
        """Check if vehicle crossed the counting line"""
        history = self.track_history.get(track_id, [])
        if len(history) < 2:
            return None

        prev_y = history[-2][1]
        curr_y = center[1]

        cooldown_frames = 30  # Prevent double counting
        if track_id in self.last_count_frame and self.frame_idx - self.last_count_frame[track_id] < cooldown_frames:
            return None

        # Moving down (entering parking)
        if prev_y < self.counting_line_y and curr_y >= self.counting_line_y:
            self.last_count_frame[track_id] = self.frame_idx
            self.vehicle_counts['in'] += 1
            self.vehicle_states[track_id] = {'direction': 'in', 'frame': self.frame_idx}
            self.save_parking_data()
            print(f"🟢 Vehicle IN - ID: {track_id}")
            return 'in'

        # Moving up (exiting parking)
        if prev_y > self.counting_line_y and curr_y <= self.counting_line_y:
            self.last_count_frame[track_id] = self.frame_idx
            self.vehicle_counts['out'] += 1
            self.vehicle_states[track_id] = {'direction': 'out', 'frame': self.frame_idx}
            self.save_parking_data()
            print(f"🔴 Vehicle OUT - ID: {track_id}")
            return 'out'

        return None

    def get_vehicle_color(self, track_id, center_y):
        """Get color for bounding box based on vehicle position"""
        line_threshold = 25
        if abs(center_y - self.counting_line_y) <= line_threshold:
            return (0, 165, 255)  # Orange for vehicles at counting line
        return (0, 255, 0)  # Green for other vehicles

    def draw_main_display(self, frame):
        """Draw the main parking information display"""
        h, w = frame.shape[:2]
        
        available, occupied = self.get_available_spaces()
        
        # Large display panel
        panel_h = 200
        panel_w = w - 40
        panel_x = 20
        panel_y = 20
        
        # Background with border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 4)
        
        # Title
        cv2.putText(frame, "🎭 THEATER PARKING", (panel_x + 20, panel_y + 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
        
        # Available spaces - Color coded
        if available > 20:
            avail_color = (0, 255, 0)  # Green - plenty of space
        elif available > 5:
            avail_color = (0, 255, 255)  # Yellow - limited space
        else:
            avail_color = (0, 0, 255)  # Red - very limited/full
        
        cv2.putText(frame, f"AVAILABLE: {available}/{self.total_capacity}", 
                    (panel_x + 20, panel_y + 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, avail_color, 3)
        
        # Status message
        if available == 0:
            status_msg = "🚫 PARKING FULL"
            status_color = (0, 0, 255)
        elif available <= 5:
            status_msg = "⚠️ ALMOST FULL"
            status_color = (0, 165, 255)
        else:
            status_msg = "✅ SPACES AVAILABLE"
            status_color = (0, 255, 0)
        
        cv2.putText(frame, status_msg, (panel_x + 20, panel_y + 140),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, status_color, 2)
        
        # Current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (panel_x + panel_w - 300, panel_y + 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    def draw_stats_panel(self, frame):
        """Draw detailed statistics panel"""
        h, w = frame.shape[:2]
        
        # Stats panel on the right
        panel_w = 300
        panel_h = 150
        panel_x = w - panel_w - 20
        panel_y = h - panel_h - 20
        
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 2)
        
        # Stats title
        cv2.putText(frame, "📊 TODAY'S STATS", (panel_x + 10, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Entry/exit counts
        cv2.putText(frame, f"Entries: {self.vehicle_counts['in']}", (panel_x + 10, panel_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Exits: {self.vehicle_counts['out']}", (panel_x + 10, panel_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        
        available, occupied = self.get_available_spaces()
        cv2.putText(frame, f"Currently: {occupied} vehicles", (panel_x + 10, panel_y + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def process_live_feed(self, camera_source=0):
        """Process live camera feed"""
        print(f"📷 Starting live camera feed from source: {camera_source}")
        
        # Open camera
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera source {camera_source}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get frame dimensions
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.setup_counting_line(h, w)
        
        print("🎬 Live processing started. Press 'q' to quit, 'f' for fullscreen, 'r' to reset counts")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading frame")
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
                    self.check_line_crossing(tid, (cx, cy))
                    
                    # Draw vehicle detection
                    color = self.get_vehicle_color(tid, cy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label
                    label = f"ID:{tid}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw all UI elements
            self.draw_counting_line(frame)
            self.draw_main_display(frame)
            self.draw_stats_panel(frame)
            
            # Display frame
            if self.display_fullscreen:
                cv2.namedWindow('Theater Parking System', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('Theater Parking System', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.namedWindow('Theater Parking System', cv2.WINDOW_NORMAL)
            
            cv2.imshow('Theater Parking System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                self.display_fullscreen = not self.display_fullscreen
            elif key == ord('r'):
                # Reset counts
                self.vehicle_counts = {'in': 0, 'out': 0}
                self.save_parking_data()
                print("🔄 Counts reset!")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Live feed stopped")

    def run_display_only(self):
        """Run display-only mode for separate monitor"""
        print("🖥️ Display-only mode started")
        
        while True:
            # Create a blank frame for display
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Draw main display
            self.draw_main_display(frame)
            self.draw_stats_panel(frame)
            
            # Show fullscreen
            cv2.namedWindow('Parking Display', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Parking Display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Parking Display', frame)
            
            key = cv2.waitKey(1000) & 0xFF  # Update every second
            if key == ord('q'):
                break
            
            # Reload data from file (in case main system updated it)
            self.load_parking_data()
        
        cv2.destroyAllWindows()


# ----------------- Main execution -----------------
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "D:\\Projects\\Vision-Based-Vehicle-Entry-Exit-Monitoring-System\\models\\my_trained_model.pt"  # Update this path
    CAMERA_SOURCE = "D:\\Projects\\Vision-Based-Vehicle-Entry-Exit-Monitoring-System\\models\\videoplayback.mp4" # 0 for default camera, or IP camera URL
    TOTAL_CAPACITY = 100  # Theater parking capacity
    
    print("🎭 Theater Parking Management System")
    print("=====================================")
    
    # Initialize system
    parking_system = TheaterParkingSystem(
        model_path=MODEL_PATH,
        total_capacity=TOTAL_CAPACITY,
        conf_thres=0.35
    )
    
    # Choose mode
    mode = input("\nChoose mode:\n1. Live Camera Processing\n2. Display Only Mode\nEnter (1 or 2): ")
    
    try:
        if mode == "1":
            parking_system.process_live_feed(CAMERA_SOURCE)
        elif mode == "2":
            import numpy as np
            parking_system.run_display_only()
        else:
            print("Invalid mode selected")
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("👋 Theater Parking System ended")