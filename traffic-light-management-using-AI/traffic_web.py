import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tracker import EuclideanDistTracker  # Ensure tracker.py exists
import matplotlib.pyplot as plt
import http.server
import socketserver
import io
import urllib.parse
import time
import uuid
import json

# Video Processing Setup
tracker = EuclideanDistTracker()
model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture('rush.mp4')

cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1000)

confThreshold = 0.4
nmsThreshold = 0.4

middle_line_position = 225
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15

classNames = model.names
required_classes = {"car": 2, "motorbike": 3, "bus": 5, "truck": 7}

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype="uint8")

temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]  # [Car, Bike, Bus, Truck]
down_list = [0, 0, 0, 0]

frame_height, frame_width = 360, 640
heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

screen_width, screen_height = 1920, 1080
grid_height, grid_width = screen_height // 2, screen_width // 2

CONGESTION_THRESHOLD = 10

VALID_USERNAME = "admin"
VALID_PASSWORD = "securepass123"

sessions = {}

# Video Processing Functions
def find_center(x, y, w, h):
    return x + w // 2, y + h // 2

def update_heatmap(cx, cy):
    if 0 <= cy < frame_height and 0 <= cx < frame_width:
        heatmap[cy, cx] += 1

def generate_heatmap():
    heatmap[:] = heatmap * 0.95
    blurred_heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    normalized_heatmap = np.uint8(255 * blurred_heatmap / (blurred_heatmap.max() + 1e-10))
    heatmap_color = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
    return cv2.resize(heatmap_color, (grid_width, grid_height))

def count_vehicle(box_id, img):
    x, y, w, h, id, index = box_id
    center = find_center(x, y, w, h)
    cx, cy = center
    update_heatmap(cx, cy)

    if up_line_position < cy < middle_line_position and id not in temp_up_list:
        temp_up_list.append(id)
    elif cy < up_line_position and id in temp_down_list:
        temp_down_list.remove(id)
        up_list[index] += 1

    if down_line_position > cy > middle_line_position and id not in temp_down_list:
        temp_down_list.append(id)
    elif cy > down_line_position and id in temp_up_list:
        temp_up_list.remove(id)
        down_list[index] += 1

    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)

def postProcess(results, img):
    detections = []
    for result in results:
        boxes = result.boxes.xywh.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for i, (box, conf, class_id) in enumerate(zip(boxes, confs, class_ids)):
            if class_id in required_classes.values() and conf > confThreshold:
                x, y, w, h = map(int, box)
                index = list(required_classes.values()).index(class_id)
                color = [int(c) for c in colors[class_id]]
                label = f"{classNames[class_id]} {int(conf * 100)}%"
                cv2.rectangle(img, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 2)
                cv2.putText(img, label, (x - w//2, y - h//2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                detections.append([x - w//2, y - h//2, w, h, index])

    tracked_objects = tracker.update(detections)
    for obj in tracked_objects:
        count_vehicle(obj, img)
    return len(tracked_objects)

def create_count_display(vehicle_count):
    count_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    cv2.putText(count_img, "Traffic Management", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(count_img, f"Up: Car-{up_list[0]} | Bike-{up_list[1]} | Bus-{up_list[2]} | Truck-{up_list[3]}", 
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(count_img, f"Down: Car-{down_list[0]} | Bike-{down_list[1]} | Bus-{down_list[2]} | Truck-{down_list[3]}", 
                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if vehicle_count > CONGESTION_THRESHOLD:
        cv2.putText(count_img, "Congestion Detected!", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(count_img, f"Vehicles: {vehicle_count}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(count_img, "Traffic Normal", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(count_img, f"Vehicles: {vehicle_count}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return count_img

def create_analytics():
    fig, ax = plt.subplots(figsize=(6, 3))
    vehicle_types = ['Car', 'Bike', 'Bus', 'Truck']
    up_counts = up_list
    down_counts = down_list
    
    x = np.arange(len(vehicle_types))
    width = 0.35
    ax.bar(x - width/2, up_counts, width, label='Up', color='g')
    ax.bar(x + width/2, down_counts, width, label='Down', color='r')
    ax.set_xticks(x)
    ax.set_xticklabels(vehicle_types)
    ax.set_title('Vehicle Analytics')
    ax.legend()
    
    canvas = fig.canvas
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    img = img.reshape((h, w, 4))[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return cv2.resize(img, (grid_width, grid_height))

def video_stream():
    while True:
        start_time = time.time()
        success, img = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        img = cv2.resize(img, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        
        results = model(img, verbose=False, imgsz=320)
        vehicle_count = postProcess(results, img)

        cv2.line(img, (0, middle_line_position), (img.shape[1], middle_line_position), (255, 0, 255), 2)
        cv2.line(img, (0, up_line_position), (img.shape[1], up_line_position), (0, 0, 255), 2)
        cv2.line(img, (0, down_line_position), (img.shape[1], down_line_position), (0, 0, 255), 2)

        video_feed = cv2.resize(img, (grid_width, grid_height))
        count_display = create_count_display(vehicle_count)
        heatmap_display = generate_heatmap()
        analytics_display = create_analytics()

        grid = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        grid[0:grid_height, 0:grid_width] = video_feed
        grid[0:grid_height, grid_width:screen_width] = count_display
        grid[grid_height:screen_height, 0:grid_width] = heatmap_display
        grid[grid_height:screen_height, grid_width:screen_width] = analytics_display

        ret, buffer = cv2.imencode('.jpg', grid, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        elapsed = time.time() - start_time
        if elapsed < 1/30:
            time.sleep(1/30 - elapsed)

# HTTP Server Handler
class TrafficWebHandler(http.server.SimpleHTTPRequestHandler):
    def get_session_token(self):
        cookie = self.headers.get('Cookie')
        if cookie:
            parts = cookie.split('=')
            if len(parts) == 2 and parts[0] == 'session_token':
                return parts[1]
        return None

    def is_authenticated(self):
        token = self.get_session_token()
        return token in sessions

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Traffic Management System</title>
                <style>
                    body { font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: linear-gradient(to right, #1e3c72, #2a5298); }
                    .login-container { background: rgba(255, 255, 255, 0.9); padding: 30px; border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.2); width: 350px; }
                    h2 { text-align: center; color: #333; }
                    input { display: block; width: 100%; margin: 15px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }
                    button { width: 100%; padding: 12px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
                    button:hover { background-color: #45a049; }
                    .error { color: #e74c3c; text-align: center; margin-top: 10px; }
                </style>
            </head>
            <body>
                <div class="login-container">
                    <h2>Traffic Management Login</h2>
                    <form method="POST" action="/login">
                        <input type="text" name="username" placeholder="Username" required>
                        <input type="password" name="password" placeholder="Password" required>
                        <button type="submit">Login</button>
                        <p class="error">{{error}}</p>
                    </form>
                </div>
            </body>
            </html>
            """.replace('{{error}}', '')
            self.wfile.write(html.encode())
        elif self.path == '/welcome':
            if not self.is_authenticated():
                self.send_response(302)
                self.send_header('Location', '/')
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Welcome - Traffic Management</title>
                <style>
                    body { font-family: Arial, sans-serif; display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh; margin: 0; background: linear-gradient(to right, #1e3c72, #2a5298); }
                    .welcome-container { background: rgba(255, 255, 255, 0.9); padding: 40px; border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.2); text-align: center; }
                    h1 { color: #333; }
                    p { color: #666; font-size: 18px; }
                    button { padding: 12px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 10px; }
                    button:hover { background-color: #45a049; }
                </style>
            </head>
            <body>
                <div class="welcome-container">
                    <h1>Welcome to Traffic Management</h1>
                    <p>Monitor and manage traffic in real-time using AI.</p>
                    <button onclick="window.location.href='/dashboard'">View Dashboard</button>
                    <button onclick="logout()">Logout</button>
                </div>
                <script>
                    function logout() {
                        document.cookie = 'session_token=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/';
                        window.location.href = '/';
                    }
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        elif self.path == '/dashboard':
            if not self.is_authenticated():
                self.send_response(302)
                self.send_header('Location', '/')
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Traffic Management Dashboard</title>
                <style>
                    body { margin: 0; font-family: Arial, sans-serif; background-color: #f0f0f0; }
                    .container { display: flex; flex-wrap: wrap; height: 100vh; width: 100vw; }
                    .box { flex: 1 1 50%; min-width: 50%; min-height: 50%; box-sizing: border-box; padding: 10px; }
                    img { width: 100%; height: 100%; object-fit: contain; }
                    .logout { position: absolute; top: 10px; right: 10px; padding: 10px; background-color: #e74c3c; color: white; border: none; border-radius: 5px; cursor: pointer; }
                    .logout:hover { background-color: #c0392b; }
                    .status { position: absolute; top: 10px; left: 10px; color: #fff; background: rgba(0, 0, 0, 0.7); padding: 5px 10px; border-radius: 5px; }
                    .analytics-btn { position: absolute; top: 50px; right: 10px; padding: 10px; background-color: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; }
                    .analytics-btn:hover { background-color: #2980b9; }
                </style>
            </head>
            <body>
                <div class="status">Logged in as: admin</div>
                <button class="logout" onclick="logout()">Logout</button>
                <button class="analytics-btn" onclick="window.location.href='/analytics'">View Analytics</button>
                <div class="container">
                    <div class="box"><img src="/video_feed" alt="Traffic Feed"></div>
                </div>
                <script>
                    function logout() {
                        document.cookie = 'session_token=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/';
                        window.location.href = '/';
                    }
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        elif self.path == '/video_feed':
            if not self.is_authenticated():
                self.send_response(302)
                self.send_header('Location', '/')
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                for frame in video_stream():
                    self.wfile.write(frame)
                    self.wfile.flush()
            except Exception as e:
                print(f"Stream interrupted: {e}")
        elif self.path == '/analytics':
            if not self.is_authenticated():
                self.send_response(302)
                self.send_header('Location', '/')
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            analytics_data = {
                'vehicle_types': ['Car', 'Bike', 'Bus', 'Truck'],
                'up_counts': up_list,
                'down_counts': down_list
            }
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Traffic Analytics</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }}
                    .chart-container {{ width: 80%; max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                    h1 {{ text-align: center; color: #333; }}
                    .back-btn {{ display: block; margin: 20px auto; padding: 10px 20px; background-color: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; }}
                    .back-btn:hover {{ background-color: #2980b9; }}
                </style>
            </head>
            <body>
                <h1>Traffic Analytics</h1>
                <div class="chart-container">
                    <canvas id="trafficChart"></canvas>
                </div>
                <button class="back-btn" onclick="window.location.href='/dashboard'">Back to Dashboard</button>
                <script>
                    const ctx = document.getElementById('trafficChart').getContext('2d');
                    const data = {json.dumps(analytics_data)};
                    new Chart(ctx, {{
                        type: 'bar',
                        data: {{
                            labels: data.vehicle_types,
                            datasets: [
                                {{
                                    label: 'Up Traffic',
                                    data: data.up_counts,
                                    backgroundColor: 'rgba(75, 192, 192, 0.7)',
                                    borderColor: 'rgba(75, 192, 192, 1)',
                                    borderWidth: 1
                                }},
                                {{
                                    label: 'Down Traffic',
                                    data: data.down_counts,
                                    backgroundColor: 'rgba(255, 99, 132, 0.7)',
                                    borderColor: 'rgba(255, 99, 132, 1)',
                                    borderWidth: 1
                                }}
                            ]
                        }},
                        options: {{
                            scales: {{
                                y: {{ beginAtZero: true, title: {{ display: true, text: 'Number of Vehicles' }} }},
                                x: {{ title: {{ display: true, text: 'Vehicle Type' }} }}
                            }},
                            plugins: {{
                                legend: {{ position: 'top' }},
                                title: {{ display: true, text: 'Vehicles Passed by Type and Direction' }}
                            }}
                        }}
                    }});
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        elif self.path == '/logout':
            token = self.get_session_token()
            if token in sessions:
                del sessions[token]
            self.send_response(302)
            self.send_header('Location', '/')
            self.send_header('Set-Cookie', 'session_token=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/')
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/login':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode()
            params = urllib.parse.parse_qs(post_data)
            username = params.get('username', [''])[0]
            password = params.get('password', [''])[0]

            if username == VALID_USERNAME and password == VALID_PASSWORD:
                session_token = str(uuid.uuid4())
                sessions[session_token] = username
                self.send_response(302)
                self.send_header('Location', '/welcome')
                self.send_header('Set-Cookie', f'session_token={session_token}; Path=/')
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Traffic Management System</title>
                    <style>
                        body { font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: linear-gradient(to right, #1e3c72, #2a5298); }
                        .login-container { background: rgba(255, 255, 255, 0.9); padding: 30px; border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.2); width: 350px; }
                        h2 { text-align: center; color: #333; }
                        input { display: block; width: 100%; margin: 15px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }
                        button { width: 100%; padding: 12px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
                        button:hover { background-color: #45a049; }
                        .error { color: #e74c3c; text-align: center; margin-top: 10px; }
                    </style>
                </head>
                <body>
                    <div class="login-container">
                        <h2>Traffic Management Login</h2>
                        <form method="POST" action="/login">
                            <input type="text" name="username" placeholder="Username" required>
                            <input type="password" name="password" placeholder="Password" required>
                            <button type="submit">Login</button>
                            <p class="error">Invalid credentials</p>
                        </form>
                    </div>
                </body>
                </html>
                """
                self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

# Run the Server
PORT = 8000
with socketserver.TCPServer(("", PORT), TrafficWebHandler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    httpd.serve_forever()