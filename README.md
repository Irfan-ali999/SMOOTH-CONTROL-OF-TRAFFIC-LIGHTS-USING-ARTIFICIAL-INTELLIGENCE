# SMOOTH-CONTROL-OF-TRAFFIC-LIGHTS-USING-ARTIFICIAL-INTELLIGENCE
Traffic Management System Documentation
1. Project Overview
1.1 Purpose
The Traffic Management System is an AI-driven solution designed to address urban traffic congestion by dynamically adapting traffic light timings based on real-time vehicle density. It uses video feeds (e.g., rush.mp4) to detect and count vehicles at traffic junctions, classifying them as cars, motorcycles, buses, or trucks. The system calculates traffic density to prioritize green signal durations for busier directions, reducing delays and improving traffic flow. It also supports emergency vehicle prioritization by providing live data for traffic signal adjustments, minimizing congestion-related risks.

1.2 Key Features
Real-Time Vehicle Detection: Uses YOLOv8 to detect and classify vehicles in video frames.
Traffic Density Calculation: Counts vehicles moving upward and downward to adjust green light timings.
Congestion Detection: Alerts when vehicle counts exceed a threshold (10 vehicles).
Heatmap Visualization: Displays traffic density using a heatmap for busy area identification.
Web Interface: Provides a user-friendly dashboard with live video, counts, heatmap, and analytics.
Analytics: Shows vehicle counts by type and direction in a bar chart.
1.3 Target Audience
Traffic managers and city planners aiming to optimize traffic flow.
Researchers studying AI-based traffic management solutions.
Developers interested in implementing intelligent transportation systems.
2. System Requirements
2.1 Software Requirements
Operating System: Windows 7+ (also compatible with macOS/Linux).
Server-side Script: Python 3.7.
IDE: PyCharm (recommended for development and debugging).
Libraries Used:
numpy: For array operations and data handling.
opencv-python: For video processing, drawing, and heatmap generation.
ultralytics: For YOLOv8 model integration.
matplotlib: For analytics visualization (bar charts).
Framework: None (uses Python’s http.server and socketserver for the web interface).
2.2 Hardware Requirements
Minimum:
CPU: Multi-core (e.g., Intel i5).
RAM: 8 GB.
Storage: 10 GB free (for Python, libraries, and video file).
Recommended:
GPU: NVIDIA GPU with CUDA support (e.g., GTX 1060) for faster YOLOv8 inference.
RAM: 16 GB.
Storage: SSD with 20 GB free.
2.3 Additional Requirements
Video File: A test video (rush.mp4) in MP4/AVI format, placed in the project directory.
Browser: Modern web browser (e.g., Chrome, Firefox) with JavaScript enabled.
3. Installation Guide
3.1 Step 1: Set Up Python Environment
Install Python 3.7 from python.org.
Create a virtual environment to manage dependencies:
bash

Copy
python -m venv traffic_env
Activate the virtual environment:
Windows: traffic_env\Scripts\activate
Linux/macOS: source traffic_env/bin/activate
3.2 Step 2: Install Required Libraries
Install the necessary libraries using pip:

bash

Copy
pip install numpy opencv-python ultralytics matplotlib
3.3 Step 3: Download YOLOv8 Model
The ultralytics library automatically downloads the YOLOv8 model (yolov8m.pt) on first run. Ensure an internet connection is available. Alternatively, download it manually from the Ultralytics GitHub and place it in the project directory.

3.4 Step 4: Prepare the Video File
Place a test video file named rush.mp4 in the project directory. The video should be in a compatible format (e.g., MP4) and can be replaced with live CCTV footage if available.

3.5 Step 5: Set Up the Tracker
Ensure the tracker.py file containing the EuclideanDistTracker class is in the project directory. This custom tracker is used to assign IDs to detected vehicles and count them across frames.

4. Usage Guide
4.1 Running the System
Save the project code in a file (e.g., traffic_system.py).
Open a terminal in the project directory and activate the virtual environment.
Run the script:
bash

Copy
python traffic_system.py
The system starts a local web server at http://localhost:8000.
4.2 Accessing the Web Interface
Open a browser and navigate to http://localhost:8000.
Log in with the default credentials:
Username: admin
Password: securepass123
After logging in, you’ll see:
Welcome Page: Options to view the dashboard or logout.
Dashboard: Displays the live video feed, vehicle counts, heatmap, and analytics.
Analytics Page: Shows a bar chart of vehicle counts by type and direction.
4.3 Interpreting the Output
Live Video Feed: Shows the video with detected vehicles (boxes drawn around them) and tracking lines (middle, upper, lower).
Vehicle Counts: Displays the number of cars, motorcycles, buses, and trucks moving upward and downward.
Heatmap: Visualizes traffic density (red for high density, blue for low).
Congestion Alert: Appears if the total vehicle count exceeds 10.
Analytics Chart: Bar chart comparing vehicle counts by type and direction.
4.4 Stopping the System
Press Ctrl+C in the terminal to stop the server.

5. Code Structure and Functionality
5.1 Main Components
Video Processing:
cv2.VideoCapture: Reads and resizes the video (rush.mp4) to 640x360 at 30 FPS.
YOLO("yolov8m.pt"): Detects vehicles with a confidence threshold of 0.4.
EuclideanDistTracker: Tracks vehicles and counts them as they cross a middle line (at 225 pixels).
Traffic Density Calculation:
Counts vehicles in up_list and down_list for each type (car, motorcycle, bus, truck).
Adjusts green signal time based on higher counts in a direction (conceptual implementation).
Heatmap and Congestion:
Uses OpenCV’s Gaussian blur to create a heatmap of traffic density.
Triggers a congestion alert if vehicle count > 10.
Web Interface:
http.server and socketserver: Hosts the web app at http://localhost:8000.
Handles routes (/, /dashboard, /video_feed, /analytics) and session management.
5.2 Key Functions
find_center(): Calculates the center of a detected vehicle for tracking.
update_heatmap(): Updates the heatmap based on vehicle positions.
count_vehicle(): Counts vehicles moving upward/downward and updates lists.
postProcess(): Processes YOLOv8 detections, draws boxes, and tracks vehicles.
create_count_display(): Generates the vehicle count display image.
create_analytics(): Creates a bar chart of vehicle counts.
video_stream(): Streams the processed video feed to the web interface.
5.3 File Structure
traffic_system.py: Main script with all code (video processing, web server).
tracker.py: Contains the EuclideanDistTracker class for vehicle tracking.
rush.mp4: Test video file for traffic monitoring.
yolov8m.pt: YOLOv8 model file (downloaded automatically).
6. Feasibility Study
6.1 Economic Feasibility
The system is low-budget, using free tools like Python, OpenCV, and YOLOv8. The web interface (http.server) requires no additional cost. Expenses are limited to a basic computer (8 GB RAM, multi-core CPU) and optional video feed licenses, ensuring affordability.

6.2 Technical Feasibility
The system has modest requirements: Python 3.7, standard libraries, and a multi-core CPU. It processes video at 640x360 resolution and 30 FPS, manageable by most systems. The web interface runs on any browser-enabled device, requiring minimal setup changes.

6.3 Social Feasibility
The system is user-friendly, accessible via a browser with minimal computer knowledge. Training helps users navigate the dashboard and interpret data (e.g., heatmap), boosting confidence and encouraging feedback from traffic managers, the end users.

7. Potential Enhancements
Live CCTV Integration: Replace rush.mp4 with live CCTV feeds for real-world deployment.
Smart Traffic Signals: Integrate with traffic lights to automatically adjust green signal times based on vehicle counts.
Predictive Analytics: Add machine learning models to predict congestion and optimize traffic flow.
Cloud Deployment: Host the system on a cloud server (e.g., AWS) for remote access.
Multi-Camera Support: Process feeds from multiple cameras to monitor larger areas.
Remove OpenCV: Replace OpenCV with imageio and Pillow for video processing and drawing, reducing dependencies.
8. Troubleshooting
Video Not Found: Ensure rush.mp4 is in the project directory. If using a different video, update the filename in the code.
Port Conflict: If http://localhost:8000 is in use, change the PORT variable in the code (e.g., to 8080).
YOLO Model Not Loading: Verify internet connectivity for automatic download, or manually place yolov8m.pt in the directory.
Slow Performance: Use a lighter YOLO model (yolov8n.pt) or reduce the frame rate in video_stream().
9. Contact and Support
For questions, contributions, or support, contact the development team at [kingirfan1sf@gmail.com]. Feedback is welcome to improve the system for future use.

Notes for Users
This documentation provides a comprehensive guide to understand, install, and use the Traffic Management System. The system’s modular design and open-source tools make it adaptable for other environments, such as different cities or traffic scenarios. Researchers can study the code to learn about AI-based traffic management, while developers can extend it for additional features like cloud support or predictive analytics.
