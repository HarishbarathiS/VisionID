from flask import Flask, request, jsonify, render_template
from face_embedding import preprocess_data, recognize_face_for_image, extract_face_embedding, SiameseNetwork
import cv2
from torchvision import transforms
import base64
import torch
from PIL import Image
import io
import numpy as np
import pandas as pd
from datetime import datetime
import os


class RecognitionResultTracker:
    def __init__(self, results_file='recognition_results.csv'):
        self.results_file = results_file
        self.initialize_results_file()
    
    def initialize_results_file(self):
        """Create results file with headers if it doesn't exist"""
        if not os.path.exists(self.results_file):
            df = pd.DataFrame(columns=[
                'recognized_name',
                'confidence_score',
                'user_verified',
                'is_correct'
            ])
            df.to_csv(self.results_file, index=False)
    
    def add_result(self, result_data):
        """Add a new result to the CSV file"""
        new_result = {
            'recognized_name': result_data['recognized_name'],
            'confidence_score': result_data['confidence_score'],
            'user_verified': True,
            'is_correct': result_data['result'] == 'yes'
        }
        
        df = pd.DataFrame([new_result])
        df.to_csv(self.results_file, mode='a', header=False, index=False)
        return new_result

app = Flask(__name__,  template_folder="../templates")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load embeddings and initialize model
embeddings_dict = preprocess_data(r'D:/HARISH/AspireNex-Tasks/Task1/face_detection_and_recognition')  # Path to stored face images
siamese_net = SiameseNetwork().to('cuda' if torch.cuda.is_available() else 'cpu')
print(embeddings_dict.keys())

@app.route('/system')
def index():
    return render_template('index.html')

# Initialize the result tracker
result_tracker = RecognitionResultTracker()

@app.route('/result', methods=['POST'])
def result():
    try:
        result_data = request.json
        
        # Add result to tracker
        saved_result = result_tracker.add_result(result_data)
        
        # Calculate current metrics
        df = pd.read_csv('recognition_results.csv')
        metrics = {
            'total_evaluations': len(df),
            'correct_recognitions': len(df[df['is_correct'] == True]),
            'accuracy': (len(df[df['is_correct'] == True]) / len(df)) * 100 if len(df) > 0 else 0,
            'average_confidence': df['confidence_score'].mean()
        }
        
        return jsonify({
            'status': 'success',
            'saved_result': saved_result,
            'current_metrics': metrics
        })
        
    except Exception as e:
        print(f"Error processing result: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/addstudent')
def add_student():
    return render_template('add_student.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/check_face', methods=['POST'])
def check_face():
    try:
        # Get image file from request
        image_file = request.files['image']
        
        # Convert to OpenCV format
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        return jsonify({
            'face_detected': len(faces) > 0
        })
        
    except Exception as e:
        print(f"Error checking face: {str(e)}")
        return jsonify({
            'face_detected': False
        })
    
@app.route("/capture-and-add", methods = ['POST'])
def capture_and_add():
    try:
        # Get image data from request
        image_data = request.json['image']
        
        # Convert base64 to image
        image_data = image_data.replace('data:image/jpeg;base64,', '')
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Face detection and recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return jsonify({
                'status': 'no_face_detected',
                'name': 'No face detected',
                'similarity_score': 0.0
            })
        
        # Process the first detected face
        x, y, w, h = faces[0]
        face_region = frame[y:y+h, x:x+w]
        
        # Preprocess face
        face_resized = cv2.resize(face_region, (160, 160))
        face_tensor = transforms.ToTensor()(face_resized).unsqueeze(0).to(torch.float32)
        face_tensor = face_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate embedding
        with torch.no_grad():
            embedding = siamese_net(face_tensor).squeeze()

        
    except : 
        print("")


@app.route('/capture', methods=['POST'])
def capture():
    try:
        # Get image data from request
        image_data = request.json['image']
        
        # Convert base64 to image
        image_data = image_data.replace('data:image/jpeg;base64,', '')
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Face detection and recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return jsonify({
                'status': 'no_face_detected',
                'name': 'No face detected',
                'similarity_score': 0.0
            })
        
        # Process the first detected face
        x, y, w, h = faces[0]
        face_region = frame[y:y+h, x:x+w]
        
        # Preprocess face
        face_resized = cv2.resize(face_region, (160, 160))
        face_tensor = transforms.ToTensor()(face_resized).unsqueeze(0).to(torch.float32)
        face_tensor = face_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate embedding
        with torch.no_grad():
            embedding = siamese_net(face_tensor).squeeze()
        
        # Recognize face
        recognized_label, similarity = recognize_face_for_image(embedding, embeddings_dict)
        
        if similarity > 0.6:
            return jsonify({
                'status': 'recognized',
                'name': recognized_label,
                'similarity_score': float(similarity)
            })
        else:
            return jsonify({
                'status': 'new_user',
                'name': 'Unknown',
                'similarity_score': float(similarity)
            })
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Load your face detection cascade and Siamese network here
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Load your siamese_net and embeddings_dict here
    
    app.run(debug=True)