import cv2
import os
import json
import time
import numpy as np
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
        
        # Color scheme - Light modern theme with accent colors
        self.colors = {
            'bg_primary': (245, 245, 250),    # Light background
            'bg_secondary': (255, 255, 255),  # White card background
            'accent_blue': (100, 150, 255),   # Blue accent
            'accent_green': (50, 200, 100),   # Success green
            'accent_red': (255, 100, 100),    # Warning red
            'accent_yellow': (255, 180, 50),  # Warning yellow
            'text_primary': (40, 40, 50),     # Dark text
            'text_secondary': (80, 80, 90),   # Gray text
            'border': (200, 200, 220),        # Light border color
            'counting_line': (0, 255, 255)    # Cyan counting line
        }
        
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

    def draw_rounded_rectangle(self, img, pt1, pt2, color, thickness, radius=15):
        """Draw rounded rectangle"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw main rectangle
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw corners
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

    def draw_gradient_background(self, frame, start_color, end_color, x1, y1, x2, y2):
        """Create gradient background effect"""
        h = y2 - y1
        for i in range(h):
            alpha = i / h
            color = tuple(int(start_color[j] * (1 - alpha) + end_color[j] * alpha) for j in range(3))
            cv2.line(frame, (x1, y1 + i), (x2, y1 + i), color, 1)

    def draw_counting_line(self, frame):
        """Draw the enhanced vehicle counting line"""
        h, w = frame.shape[:2]
        
        # Draw glowing effect for counting line
        for thickness in [8, 6, 4, 2]:
            alpha = 0.3 if thickness == 8 else 0.5 if thickness == 6 else 0.7 if thickness == 4 else 1.0
            line_color = tuple(int(c * alpha) for c in self.colors['counting_line'])
            cv2.line(frame, (0, self.counting_line_y), (w, self.counting_line_y), line_color, thickness)
        
        # Counting line label with modern styling
        label_bg_w, label_bg_h = 160, 30
        label_x = 20
        label_y = self.counting_line_y - 45
        
        # Background for label
        overlay = frame.copy()
        cv2.rectangle(overlay, (label_x, label_y), (label_x + label_bg_w, label_y + label_bg_h), 
                     self.colors['bg_secondary'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        cv2.rectangle(frame, (label_x, label_y), (label_x + label_bg_w, label_y + label_bg_h), 
                     self.colors['counting_line'], 2)
        
        # Text
        cv2.putText(frame, "COUNTING LINE", (label_x + 8, label_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_primary'], 1)

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
        line_threshold = 30
        if abs(center_y - self.counting_line_y) <= line_threshold:
            return self.colors['accent_yellow']  # Yellow for vehicles at counting line
        return self.colors['accent_green']  # Green for other vehicles

    def draw_main_display(self, frame):
        """Draw the compact main parking information display"""
        h, w = frame.shape[:2]
        available, occupied = self.get_available_spaces()
        
        # Compact header panel
        header_h = 60
        header_w = w - 40
        header_x = 20
        header_y = 10
        
        # Light background
        overlay = frame.copy()
        cv2.rectangle(overlay, (header_x, header_y), (header_x + header_w, header_y + header_h), 
                     self.colors['bg_secondary'], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Border
        cv2.rectangle(frame, (header_x, header_y), (header_x + header_w, header_y + header_h), 
                     self.colors['border'], 2)
        
        # Title - more compact
        title_y = header_y + 25
        cv2.putText(frame, "VEHICLE IN-OUT MANAGEMENT", (header_x + 20, title_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, self.colors['text_primary'], 2)
        
        # Subtitle - smaller
        cv2.putText(frame, "Real-time Monitoring", (header_x + 20, title_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_secondary'], 1)
        
        # Current time - top right, smaller
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_x = header_x + header_w - 250
        cv2.putText(frame, current_time, (time_x, title_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['accent_blue'], 1)

    def draw_status_cards(self, frame):
        """Draw compact status cards for parking information"""
        h, w = frame.shape[:2]
        available, occupied = self.get_available_spaces()
        
        # Smaller card dimensions
        card_w = 200
        card_h = 80
        card_spacing = 25
        cards_start_y = 85
        
        # Calculate positions for 3 cards
        total_width = (card_w * 3) + (card_spacing * 2)
        start_x = (w - total_width) // 2
        
        cards = [
            {
                'title': 'AVAILABLE',
                'value': f"{available}",
                'subtitle': f"of {self.total_capacity}",
                'color': self.colors['accent_green'] if available > 20 else 
                        self.colors['accent_yellow'] if available > 5 else self.colors['accent_red'],
                'x': start_x
            },
            {
                'title': 'IN TODAY',
                'value': f"{self.vehicle_counts['in']}",
                'subtitle': 'entries',
                'color': self.colors['accent_green'],
                'x': start_x + card_w + card_spacing
            },
            {
                'title': 'OUT TODAY', 
                'value': f"{self.vehicle_counts['out']}",
                'subtitle': 'exits',
                'color': self.colors['accent_blue'],
                'x': start_x + (card_w + card_spacing) * 2
            }
        ]
        
        for card in cards:
            x, y = card['x'], cards_start_y
            
            # Card background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + card_w, y + card_h), self.colors['bg_secondary'], -1)
            cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
            
            # Card border
            cv2.rectangle(frame, (x, y), (x + card_w, y + card_h), card['color'], 2)
            
            # Top accent line - thinner
            cv2.rectangle(frame, (x, y), (x + card_w, y + 3), card['color'], -1)
            
            # Card content - more compact
            # Title - smaller
            cv2.putText(frame, card['title'], (x + 8, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_secondary'], 1)
            
            # Main value - smaller but prominent
            cv2.putText(frame, card['value'], (x + 8, y + 50),
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, card['color'], 2)
            
            # Subtitle - smaller
            cv2.putText(frame, card['subtitle'], (x + 8, y + 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors['text_secondary'], 1)

    def draw_status_message(self, frame):
        """Draw compact current parking status message"""
        h, w = frame.shape[:2]
        available, occupied = self.get_available_spaces()
        
        # Status message area - smaller
        msg_y = 180
        msg_h = 35
        
        if available == 0:
            status_msg = "PARKING FULL - NO SPACES AVAILABLE"
            status_color = self.colors['accent_red']
            bg_color = (255, 240, 240)
        elif available <= 5:
            status_msg = "LIMITED SPACES - ALMOST FULL"
            status_color = self.colors['accent_yellow']
            bg_color = (255, 250, 240)
        elif available <= 20:
            status_msg = "MODERATE AVAILABILITY"
            status_color = self.colors['accent_blue']
            bg_color = (240, 245, 255)
        else:
            status_msg = "PLENTY OF SPACES AVAILABLE"
            status_color = self.colors['accent_green']
            bg_color = (240, 255, 245)
        
        # Light background
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, msg_y), (w - 20, msg_y + msg_h), bg_color, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        cv2.rectangle(frame, (20, msg_y), (w - 20, msg_y + msg_h), status_color, 2)
        
        # Message text (centered) - smaller
        text_size = cv2.getTextSize(status_msg, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, status_msg, (text_x, msg_y + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 2)

    def draw_live_stats_panel(self, frame):
        """Draw compact live statistics panel"""
        h, w = frame.shape[:2]
        available, occupied = self.get_available_spaces()
        
        # Smaller panel position (bottom right)
        panel_w = 250
        panel_h = 120
        panel_x = w - panel_w - 15
        panel_y = h - panel_h - 15
        
        # Light background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     self.colors['bg_secondary'], -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     self.colors['accent_blue'], 2)
        
        # Header - smaller
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + 25), 
                     self.colors['accent_blue'], -1)
        cv2.putText(frame, "LIVE STATISTICS", (panel_x + 10, panel_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Stats content - more compact
        stats_y = panel_y + 40
        line_height = 20
        
        stats = [
            f"Entries: {self.vehicle_counts['in']}",
            f"Exits: {self.vehicle_counts['out']}",
            f"Inside: {occupied}",
            f"Rate: {(occupied/self.total_capacity)*100:.1f}%"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (panel_x + 10, stats_y + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['text_primary'], 1)

    def process_live_feed(self, camera_source=0):
        """Process live camera feed with enhanced UI"""
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
        
        print("🎬 Live processing started. Controls:")
        print("   'q' - Quit")
        print("   'f' - Toggle fullscreen")
        print("   'r' - Reset counts")
        
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
                    direction = self.check_line_crossing(tid, (cx, cy))
                    
                    # Enhanced vehicle detection drawing
                    color = self.get_vehicle_color(tid, cy)
                    
                    # Draw bounding box with thicker lines
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw filled corner indicators
                    corner_size = 15
                    cv2.rectangle(frame, (x1, y1), (x1 + corner_size, y1 + corner_size), color, -1)
                    cv2.rectangle(frame, (x2 - corner_size, y1), (x2, y1 + corner_size), color, -1)
                    cv2.rectangle(frame, (x1, y2 - corner_size), (x1 + corner_size, y2), color, -1)
                    cv2.rectangle(frame, (x2 - corner_size, y2 - corner_size), (x2, y2), color, -1)
                    
                    # Enhanced label background
                    label = f"ID:{tid}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 10), 
                              cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Show direction indicator if just crossed
                    if direction:
                        indicator = "↓ IN" if direction == 'in' else "↑ OUT"
                        cv2.putText(frame, indicator, (cx - 30, cy - 20), 
                                  cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            
            # Draw all UI elements
            self.draw_counting_line(frame)
            self.draw_main_display(frame)
            self.draw_status_cards(frame)
            self.draw_status_message(frame)
            self.draw_live_stats_panel(frame)
            
            # Display frame
            if self.display_fullscreen:
                cv2.namedWindow('Theater Parking System', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('Theater Parking System', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.namedWindow('Theater Parking System', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Theater Parking System', 1280, 720)
            
            cv2.imshow('Theater Parking System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                self.display_fullscreen = not self.display_fullscreen
            elif key == ord('r'):
                # Reset counts with confirmation
                self.vehicle_counts = {'in': 0, 'out': 0}
                self.save_parking_data()
                print("🔄 Counts reset!")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Live feed stopped")

    def run_display_only(self):
        """Run enhanced display-only mode for separate monitor"""
        print("🖥️ Enhanced display-only mode started")
        
        while True:
            # Create a blank frame for display
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Draw all UI elements
            self.draw_main_display(frame)
            self.draw_status_cards(frame) 
            self.draw_status_message(frame)
            self.draw_live_stats_panel(frame)
            
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
    MODEL_PATH = "D:\\Projects\\Vision-Based-Vehicle-Entry-Exit-Monitoring-System\\models\\my_trained_model.pt"
    CAMERA_SOURCE = "D:\\Projects\\Vision-Based-Vehicle-Entry-Exit-Monitoring-System\\models\\videoplayback.mp4"
    TOTAL_CAPACITY = 100  # Theater parking capacity
    
    print("🚗 Enhanced Vehicle In-Out Management System")
    print("==========================================")
    print("✨ Features:")
    print("   • Clean light theme UI")
    print("   • Compact layout design")
    print("   • Real-time vehicle tracking") 
    print("   • Visual status indicators")
    print("   • Live statistics")
    print("   • Minimalist interface")
    
    # Initialize system
    parking_system = TheaterParkingSystem(
        model_path=MODEL_PATH,
        total_capacity=TOTAL_CAPACITY,
        conf_thres=0.35
    )
    
    # Choose mode
    print("\n🎯 Choose operation mode:")
    print("1. Live Camera Processing (Full system)")
    print("2. Display Only Mode (Statistics display)")
    mode = input("Enter your choice (1 or 2): ")
    
    try:
        if mode == "1":
            parking_system.process_live_feed(CAMERA_SOURCE)
        elif mode == "2":
            parking_system.run_display_only()
        else:
            print("❌ Invalid mode selected")
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    
        print("👋 Vehicle In-Out Management System ended")