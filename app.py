import json
import os
import cv2
import dlib
import torch
import numpy as np
import pyttsx3
import pytesseract
import psutil
import queue
import threading
from flask import Flask, render_template, Response,jsonify,request
from scipy.spatial.distance import euclidean
from gtts import gTTS
import pygame
import os
import time
import geocoder
import warnings
import requests
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask import redirect, url_for
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a random secret key

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    location_history = db.relationship('LocationHistory', backref='user', lazy=True)
# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLOv5s model for object detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to('cpu')

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
speech_queue = queue.Queue()


def preprocess_for_text_detection(frame):
    """Preprocess image for better text detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    # Using adaptive thresholding instead of simple thresholding
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return binary



def _run_speech():
    """Background speech processing thread."""
    while True:
        text = speech_queue.get()
        if text is None:
            break  # Exit thread if None received

        try:
            print(f"Speaking: {text}")  # Debugging output
            engine.say(text)
            engine.runAndWait()
        except RuntimeError as e:
            print(f"Speech error: {e}")

# Start background speech thread
speech_thread = threading.Thread(target=_run_speech, daemon=True)
speech_thread.start()

# def speak(text):
#     engine = pyttsx3.init()
#     engine.say(text)
#     engine.runAndWait()
def speak(text):
    """Queue text for speech processing."""
    speech_queue.put(text)
threading.Thread(target=speak, args=("Object detected",)).start()


def _run_speech(text):
    """Function to run text-to-speech processing."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    except RuntimeError as e:
        print(f"Speech error: {e}")

# Background thread for speech processing
def speech_worker():
    """Continuously process text-to-speech requests from the queue."""
    while True:
        text = speech_queue.get()
        if text is None:  # Exit condition
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except RuntimeError as e:
            print(f"Speech error: {e}")

# Start speech thread once
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()


def speak_text(text):
    """Generate speech using gTTS and play it using pygame."""
    try:
        tts = gTTS(text=text, lang="en")
        filename = "temp_audio.mp3"
        tts.save(filename)

        pygame.mixer.init()  # Initialize mixer
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():  # Wait until the audio finishes playing
            time.sleep(0.1)

        pygame.mixer.quit()  # Ensure pygame quits properly
        os.remove(filename)  # Remove file after playing
    except Exception as e:
        print(f"Speech error: {e}")


# Initialize the face recognition model (dlib)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Known face encodings and labels
known_face_encodings = []
known_face_names = []
known_faces_dir = "known_faces"

# Constants
FRAME_SKIP = 1
TEXT_DETECTION_SKIP = FRAME_SKIP * 0
RESIZE_FACTOR = 0.1

# Environment detection
ENVIRONMENTS = {
    "street": ["car", "traffic light", "bus", "truck"],
    "park": ["tree", "bench", "dog", "bird"],
    "indoors": ["person", "chair", "table", "tv", "sofa"]
}
current_environment = None
last_battery_announcement = None



def detect_environment(labels):
    """Determine the likely environment based on detected objects."""
    global current_environment
    environment_counts = {env: 0 for env in ENVIRONMENTS}
    
    for label in labels:
        for env, objects in ENVIRONMENTS.items():
            if label in objects:
                environment_counts[env] += 1

    detected_environment = max(environment_counts, key=environment_counts.get)
    
    if environment_counts[detected_environment] > 0 and detected_environment != current_environment:
        current_environment = detected_environment
        speak_text(f"Environment detected: {current_environment}")
        print(f"Environment detected: {current_environment}")
        provide_environment_feedback(current_environment)

def provide_environment_feedback(environment):
    """Provide specific guidance based on the detected environment."""
    messages = {
        "street": "You are near a street. Watch out for vehicles and traffic lights.",
        "park": "You are in a park. Enjoy the surroundings but be aware of obstacles like benches and trees.",
        "indoors": "You are indoors. Navigate carefully around furniture and people."
    }
    speak_text(messages.get(environment, ""))

# import time

last_speech_time = {}

def detect_objects(frame):
    """Detect objects, draw bounding boxes, and provide audio feedback."""
    global last_speech_time
    small_frame = cv2.resize(frame, (640, 640))  # Resize to 640x640
    small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

    try:
        results = model([small_frame_rgb])  # Perform object detection
    except Exception as e:
        print(f"Error with YOLOv5 model: {e}")
        return [], frame

    detections = results.pandas().xyxy[0]
    
    if detections.empty:
        return [], frame

    labels = []
    current_time = time.time()
    confidence_threshold = 0.5
    obstacle_objects = ["tv", "car", "truck", "aeroplane"]  # Objects to treat as obstacles

    for _, row in detections.iterrows():
        if row['confidence'] >= confidence_threshold:
            class_name = row['name']
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])

            # Adjust bounding box coordinates to original frame size
            x1, y1, x2, y2 = int(x1 * (frame.shape[1] / 640)), int(y1 * (frame.shape[0] / 640)), \
                              int(x2 * (frame.shape[1] / 640)), int(y2 * (frame.shape[0] / 640))

            labels.append(class_name)

            # Draw bounding box on original frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Avoid repeating speech too frequently
            if class_name not in last_speech_time or (current_time - last_speech_time[class_name] > 5):
                last_speech_time[class_name] = current_time
                
                # Check if the object is an obstacle
                if class_name.lower() in obstacle_objects:
                    speak_text(f"Obstacle detected: {class_name}")
                else:
                    speak_text(f"Detected {class_name}")

    detect_environment(labels)
    return labels, frame

def detect_text(frame):
    """Detect text in the frame and provide audio feedback."""
    processed_frame = preprocess_for_text_detection(frame)
    text = pytesseract.image_to_string(processed_frame, config='--psm 6')  # PSM 6 for block text
    
    if text.strip():
        print("Detected text:", text)  # Debugging output
        speak_text(f"Text detected: {text}")
    else:
        print("No text detected in this frame.")  # Debugging output for when no text is found

    cv2.imshow('Processed Frame', processed_frame)
    cv2.waitKey(1)


def check_battery():
    """Check battery level and provide alert if low."""
    global last_battery_announcement
    battery = psutil.sensors_battery()
    if battery and battery.percent % 10 == 0 and battery.percent != last_battery_announcement:
        last_battery_announcement = battery.percent
        speak_text(f"Battery level at {battery.percent} percent.")
        if battery.percent < 20 and not battery.power_plugged:
            speak_text("Battery low, please charge your device.")

def add_known_face(face_image, name):
    """Add a known face to the system."""
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        face_encoding = np.array(face_rec_model.compute_face_descriptor(face_image, shape))
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)

# Load known faces
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg"):
        face_image = cv2.imread(os.path.join(known_faces_dir, filename))
        name = os.path.splitext(filename)[0]
        add_known_face(face_image, name)

last_spoken_time = {}

def recognize_faces(frame):
    """Detect and recognize known faces."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        face_encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape))

        matches = [euclidean(face_encoding, known) < 0.5 for known in known_face_encodings]
        name = "Unknown"

        if True in matches:
            name = known_face_names[matches.index(True)]

        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if name in last_spoken_time and current_time - last_spoken_time[name] < 5:
            continue  # Avoid frequent speech

        if name != "Unknown":
            last_spoken_time[name] = current_time
            speak_text(f"This is {name}")

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def gen_frames():
    """Capture and process video frames."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        labels, frame = detect_objects(frame)
        recognize_faces(frame)

        if cv2.getTickCount() % (cv2.getTickFrequency() * 3) == 0:
            detect_text(frame)

        check_battery()

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    if current_user.is_authenticated:
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/speak')
def trigger_speech():
    threading.Thread(target=speak, args=("Object detected",)).start()
    return "Speaking..."

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/location', methods=['GET'])
def get_location():
    """Fetch user's approximate location using an external API"""
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        if 'loc' in data:
            lat, lon = map(float, data['loc'].split(','))
            print(f"Location fetched: Latitude: {lat}, Longitude: {lon}")  # Debugging output
            return jsonify({'latitude': lat, 'longitude': lon})
    except Exception as e:
        print(f"Error fetching location: {e}")
    
    return jsonify({'error': 'Unable to get location'}), 500



@app.route('/track_location')
def track_location():
    return render_template('loc.html')


@app.route('/update_location', methods=['POST'])
@login_required
def update_location():
    data = request.get_json()
    
    # Store location in history (only if 30 minutes have passed since last record)
    last_location = LocationHistory.query.filter_by(user_id=current_user.id)\
        .order_by(LocationHistory.timestamp.desc()).first()
    
    now = datetime.utcnow()
    if not last_location or (now - last_location.timestamp).total_seconds() >= 1800:  # 30 minutes
        new_location = LocationHistory(
            user_id=current_user.id,
            latitude=data['latitude'],
            longitude=data['longitude'],
            timestamp=now
        )
        db.session.add(new_location)
        db.session.commit()
    
    return jsonify({"status": "Location updated"}), 200

def location_generator():
    """Continuously fetch and stream location data"""
    while True:
        try:
            response = requests.get('https://ipinfo.io/json')
            data = response.json()
            if 'loc' in data:
                lat, lon = map(float, data['loc'].split(','))
                latest_location = f"Latitude: {lat}, Longitude: {lon}"
                print(f"Streaming location: {latest_location}")  # Debugging output
                yield f"data: {latest_location}\n\n"
            else:
                yield "data: Unable to get location\n\n"
        except Exception as e:
            print(f"Error fetching location: {e}")
            yield "data: Error fetching location\n\n"

        time.sleep(5)  # Update location every 5 seconds
        
# This will store the current location, could be updated by other processes
current_location = {"latitude": 18.5196, "longitude": 73.8554}

@app.route('/location_stream')
def location_stream():
    def generate_location():
        while True:
            # This is where the location would be dynamically updated
            # For demonstration, we can send the same location
            yield f"data: {json.dumps(current_location)}\n\n"
            time.sleep(1)  # Update every second
    
    return Response(generate_location(), content_type='text/event-stream')

@app.route('/location_history')
@login_required
def location_history():
    history = LocationHistory.query.filter_by(user_id=current_user.id)\
        .order_by(LocationHistory.timestamp.desc()).all()
    return jsonify([{
        'latitude': loc.latitude,
        'longitude': loc.longitude,
        'timestamp': loc.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    } for loc in history])

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        
        return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already exists')
        
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error='Email already exists')
        
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        return redirect(url_for('index'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

class LocationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)