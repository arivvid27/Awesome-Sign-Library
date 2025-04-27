from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import time
import google.generativeai as genai
from gtts import gTTS
import json
import string
import threading
import base64

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Load the ASL recognition model (we'll assume you've trained it)
# This is a placeholder - you'll need to train and save your own model
try:
    asl_model = tf.keras.models.load_model('models/letter_model.h5')
    model_loaded = True
except:
    model_loaded = False
    print("Warning: ASL model not found. App will run in demo mode.")

# Configure Google Gemini API
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyCfb1AlLLYl9V3gEODD1JKwsuLTqQi0E3Q")
    genai.configure(api_key=GOOGLE_API_KEY)
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 64,
    }
    gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite", generation_config=generation_config)
    gemini_available = True
except:
    gemini_available = False
    print("Warning: Google Gemini API configuration failed. Word segmentation will be limited.")

# Global variables
detected_letters = []
current_mode = "letter"  # Default mode
last_detection_time = time.time()
detection_cooldown = 1.0  # seconds between detections
speech_thread = None
last_spoken_text = ""

# ASL alphabet for recognition
asl_classes = list(string.ascii_uppercase) + ["SPACE", "DELETE"]

def demo_predict_asl(hand_landmarks):
    """
    Demo function that returns random ASL letters for demonstration
    when no model is available
    """
    import random
    # In a real implementation, this would use the hand landmarks for classification
    # This is just a placeholder for demonstration
    return random.choice(asl_classes)

def predict_asl(frame, hand_landmarks):
    """
    Use the trained model to predict the ASL sign
    """
    if not model_loaded:
        return demo_predict_asl(hand_landmarks)
    
    # Extract hand features from landmarks
    # This is a simplified example - you would need to process landmarks properly
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    
    # Normalize features
    features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = asl_model.predict(features)
    predicted_class = np.argmax(prediction)
    
    return asl_classes[predicted_class]

def segment_text_with_gemini(text):
    """
    Use Google Gemini to segment a string of ASL letters into words
    """
    if not gemini_available or not text:
        return text
    
    try:
        prompt = f"""
        I have a string of letters from ASL recognition that needs to be segmented into words.
        There are no spaces in ASL, so I need you to add spaces where words likely begin and end.
        
        Here is the string of letters: {text}
        
        Please return only the segmented text with spaces. Do not include any other explanation.
        """
        
        response = gemini_model.generate_content(prompt)
        segmented_text = response.text.strip()
        return segmented_text
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return text

def text_to_speech(text):
    """
    Convert text to speech using gTTS
    """
    global last_spoken_text
    
    if text == last_spoken_text:
        return
    
    last_spoken_text = text
    
    try:
        tts = gTTS(text=text, lang='en')
        speech_file = "static/temp_speech.mp3"
        tts.save(speech_file)
        
        # Use playsound to play the audio
        os.system(f"mpg123 {speech_file}")
    except Exception as e:
        print(f"TTS Error: {e}")

def speak_in_background(text):
    """
    Start speech in a separate thread
    """
    global speech_thread
    
    if speech_thread is not None and speech_thread.is_alive():
        return
    
    speech_thread = threading.Thread(target=text_to_speech, args=(text,))
    speech_thread.daemon = True
    speech_thread.start()

def process_frame(frame):
    """
    Process each video frame to detect and recognize ASL signs
    """
    global detected_letters, last_detection_time
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Only detect a new sign after cooldown period
            current_time = time.time()
            if current_time - last_detection_time >= detection_cooldown:
                # Predict ASL sign
                predicted_letter = predict_asl(rgb_frame, hand_landmarks)
                
                # Handle special characters
                if predicted_letter == "DELETE" and detected_letters:
                    detected_letters.pop()
                elif predicted_letter == "SPACE":
                    detected_letters.append(" ")
                elif predicted_letter in asl_classes:
                    detected_letters.append(predicted_letter)
                
                # Speak the detected letter or word
                if current_mode == "letter":
                    speak_in_background(predicted_letter)
                else:  # word mode
                    # Wait for a pause to detect a complete word
                    # This is simplified - a more advanced approach would be better
                    if predicted_letter == "SPACE":
                        text = ''.join(detected_letters[-10:])  # Last few letters
                        speak_in_background(text.strip())
                
                last_detection_time = current_time
    
    return frame

def generate_frames():
    """
    Generate frames from the webcam
    """
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process the frame
            processed_frame = process_frame(frame)
            
            # Add text overlay showing detected letters
            text = ''.join(detected_letters[-20:])  # Show last 20 detected letters
            cv2.putText(processed_frame, text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add mode indicator
            mode_text = f"Mode: {current_mode.capitalize()}"
            cv2.putText(processed_frame, mode_text, (10, frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Encode the frame
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """
    Render the main page
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    """
    Set the recognition mode (letter or word)
    """
    global current_mode
    data = request.get_json()
    current_mode = data.get('mode', 'letter')
    return jsonify({"status": "success", "mode": current_mode})

@app.route('/get_text')
def get_text():
    """
    Get the current detected text
    """
    global detected_letters
    raw_text = ''.join(detected_letters)
    
    if current_mode == "word":
        # Use Gemini to segment the text into words
        segmented_text = segment_text_with_gemini(raw_text)
        return jsonify({
            "raw_text": raw_text,
            "segmented_text": segmented_text
        })
    else:
        return jsonify({
            "raw_text": raw_text,
            "segmented_text": raw_text
        })

@app.route('/clear_text', methods=['POST'])
def clear_text():
    """
    Clear the detected text
    """
    global detected_letters
    detected_letters = []
    return jsonify({"status": "success"})

@app.route('/speak_text', methods=['POST'])
def speak_text():
    """
    Speak the current text
    """
    data = request.get_json()
    text = data.get('text', '')
    speak_in_background(text)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True)