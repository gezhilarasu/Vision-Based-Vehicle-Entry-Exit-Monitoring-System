# 🚗 Vision-Based Vehicle Entry-Exit Monitoring System

An AI-powered vehicle tracking and counting system that monitors vehicle movement across designated entry/exit points using computer vision and YOLO object detection.

## 📋 Table of Contents
- [Overview](#overview)
- [Real-World Applications](#real-world-applications)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Results & Output](#results--output)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This system uses advanced computer vision techniques to automatically detect, track, and count vehicles crossing predefined boundaries in video footage. It's designed for real-time monitoring applications where accurate vehicle counting and traffic analysis are essential.

### Key Capabilities
- **Real-time vehicle detection** using YOLO (You Only Look Once) deep learning model
- **Multi-object tracking** with ByteTrack algorithm
- **Bidirectional counting** (entry and exit detection)
- **Vehicle type classification** (cars, trucks, buses, motorcycles, etc.)
- **Visual analytics dashboard** with live statistics
- **Export functionality** for processed videos and CSV reports

## 🌍 Real-World Applications

### 🅿️ Parking Management Systems
- **Mall Parking**: Track available spots in real-time, display "Parking Full" when capacity reached
- **Hospital Parking**: Monitor visitor vs. staff parking areas separately
- **Airport Parking**: Manage different parking zones (short-term, long-term, premium)
- **University Campus**: Control access during peak hours and events

### 🏭 Industrial & Commercial
- **Factory Gates**: Monitor employee and visitor vehicle access
- **Warehouse Loading Docks**: Track delivery truck arrivals and departures
- **Construction Sites**: Control contractor vehicle access for security
- **Shopping Centers**: Analyze peak traffic hours for staffing optimization

### 🚦 Traffic Management
- **Highway Toll Plazas**: Automated vehicle counting for traffic analysis
- **City Traffic Monitoring**: Understand traffic patterns for urban planning
- **Bridge/Tunnel Monitoring**: Track vehicle flow for maintenance scheduling
- **Event Traffic Control**: Monitor vehicle flow during concerts, sports events

### 🏢 Access Control Systems
- **Residential Complexes**: Monitor visitor vehicles and resident access
- **Corporate Offices**: Track employee parking usage and optimize space
- **Government Facilities**: Enhanced security through vehicle monitoring
- **Hotel Parking**: Manage guest parking and valet services

### 📊 Business Intelligence
- **Retail Analytics**: Correlate foot traffic with vehicle arrivals
- **Peak Hour Analysis**: Optimize staffing based on traffic patterns
- **Capacity Planning**: Historical data for parking expansion decisions
- **Revenue Optimization**: Dynamic pricing based on demand patterns

## ✨ Features

### Core Functionality
- ✅ **Automated Vehicle Detection**: Identifies cars, trucks, buses, motorcycles, and more
- ✅ **Intelligent Tracking**: Maintains vehicle identity across video frames
- ✅ **Bidirectional Counting**: Separate counters for incoming and outgoing vehicles
- ✅ **Real-time Processing**: Live video analysis with minimal latency
- ✅ **Visual Feedback**: Color-coded bounding boxes and movement indicators

### Advanced Analytics
- 📊 **Vehicle Classification**: Breakdown by vehicle type (sedan, SUV, truck, etc.)
- 📈 **Traffic Statistics**: Total counts, peak hours, average dwell time
- 🎯 **Accuracy Optimization**: Configurable confidence thresholds
- 📱 **Web Interface**: User-friendly Streamlit dashboard

### Export & Integration
- 💾 **Video Export**: Download processed videos with annotations
- 📋 **CSV Reports**: Detailed analytics in spreadsheet format
- 🔄 **API Ready**: Easy integration with existing systems
- ⚙️ **Configurable**: Adjustable parameters for different environments

## 🛠 Technology Stack

### AI & Computer Vision
- **YOLO (Ultralytics)**: State-of-the-art object detection
- **ByteTrack**: Multi-object tracking algorithm
- **OpenCV**: Computer vision operations and video processing
- **NumPy**: Numerical computing for image arrays

### Web Framework & UI
- **Streamlit**: Interactive web application framework
- **Pandas**: Data manipulation and analysis
- **Python 3.8+**: Core programming language

### Deployment
- **Streamlit Cloud**: Cloud deployment ready
- **Docker**: Containerization support (optional)
- **AWS/GCP**: Cloud infrastructure compatible

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Git
- Webcam or video files for testing

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/vision-based-vehicle-entry-exit-monitoring-system.git
cd vision-based-vehicle-entry-exit-monitoring-system
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download YOLO Model
```bash
# Create models directory
mkdir models

# Download pre-trained model (or place your custom model)
# Place your trained model as: models/my_trained_model.pt
```

### Step 5: Run Application
```bash
streamlit run app.py
```

## 🚀 Usage

### Basic Operation
1. **Launch Application**: Run `streamlit run app.py`
2. **Upload Video**: Use the file uploader to select your video file
3. **Process Video**: Click "Process Video" to start analysis
4. **View Results**: Monitor real-time statistics and vehicle counts
5. **Download**: Export processed video and CSV report

### Supported Video Formats
- MP4 (recommended)
- AVI
- MOV
- MKV

### Configuration Options
- **Confidence Threshold**: Adjust detection sensitivity (0.1 - 1.0)
- **Counting Line Position**: Modify via code (default: 50% of frame height)
- **Vehicle Classes**: Customize which object types to count

## 📁 Project Structure

```
vehicle-monitoring-system/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── packages.txt          # System packages (for Streamlit Cloud)
├── models/               # YOLO model files
│   └── my_trained_model.pt
├── sample_videos/        # Test video files
├── exports/              # Generated reports and videos
├── README.md             # Project documentation
└── .gitignore           # Git ignore rules
```

## ⚙️ How It Works

### 1. Video Input Processing
```python
# Video is loaded and processed frame by frame
cap = cv2.VideoCapture(video_path)
```

### 2. Object Detection
```python
# YOLO detects vehicles in each frame
results = self.model.track(frame, persist=True, conf=self.conf_thres)
```

### 3. Multi-Object Tracking
- ByteTrack algorithm maintains vehicle identity across frames
- Each vehicle gets a unique tracking ID
- Historical positions are stored for movement analysis

### 4. Counting Logic
```python
# Check if vehicle crossed the counting line
if prev_y < self.counting_line_y and curr_y >= self.counting_line_y:
    self.vehicle_counts['in'] += 1  # Vehicle entering
elif prev_y > self.counting_line_y and curr_y <= self.counting_line_y:
    self.vehicle_counts['out'] += 1  # Vehicle exiting
```

### 5. Visualization
- Bounding boxes around detected vehicles
- Movement direction indicators
- Real-time statistics overlay
- Counting line visualization

## 📊 Results & Output

### Real-time Dashboard
- **Live Vehicle Count**: Current vehicles in/out/total
- **Vehicle Type Breakdown**: Classification by vehicle category
- **Visual Indicators**: Color-coded tracking boxes and direction arrows
- **Processing Status**: Real-time progress updates

### Export Options
1. **Processed Video**: MP4 file with all annotations and statistics
2. **CSV Report**: Detailed analytics including:
   - Total vehicle counts
   - Vehicle type distribution
   - Timestamp data
   - Processing metadata

### Sample Output
```
Metric,Count
Total Vehicles,127
Vehicles In,64
Vehicles Out,63
Car Count,89
Truck Count,23
Bus Count,8
Motorcycle Count,7
```

## 🎛 Configuration

### Model Parameters
```python
# In VehicleManagementSystem.__init__()
conf_thres=0.35          # Detection confidence threshold
min_conf_for_draw=0.5    # Minimum confidence for display
```

### Counting Line
```python
# Adjust counting line position (0.0 to 1.0)
ratio = 0.5  # 50% of frame height
```

### Vehicle Classes
```python
# Customize vehicle types to detect
vehicle_keywords = ['car', 'truck', 'bus', 'motor', 'bike', 'van']
```

## 🔧 Troubleshooting

### Common Issues

**1. Model Not Found**
```bash
Error: Model file not found at: models/my_trained_model.pt
```
Solution: Ensure YOLO model file exists in the models directory

**2. OpenCV Import Error**
```bash
ImportError: libGL.so.1: cannot open shared object file
```
Solution: Use `opencv-python-headless` in requirements.txt

**3. Memory Issues**
```bash
RuntimeError: CUDA out of memory
```
Solution: Reduce video resolution or process shorter clips

### Performance Optimization
- Use GPU acceleration if available
- Reduce input video resolution for faster processing
- Adjust confidence thresholds to reduce false detections
- Process shorter video segments for testing

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the Repository**
2. **Create Feature Branch**: `git checkout -b feature/your-feature`
3. **Commit Changes**: `git commit -m "Add your feature"`
4. **Push to Branch**: `git push origin feature/your-feature`
5. **Create Pull Request**

### Areas for Contribution
- Enhanced vehicle classification models
- Real-time camera integration
- Mobile app development
- Advanced analytics features
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* -G EZHILARASU

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://ultralytics.com/) for the object detection framework
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for multi-object tracking
- [Streamlit](https://streamlit.io/) for the web application framework
- OpenCV community for computer vision tools

## 📞 Support

For support, email your-email@example.com or create an issue in this repository.

---

**⭐ Star this repository if you found it helpful!**