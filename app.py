import os
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import json
from asl_detector import ASLDetector
from text_processor import TextProcessor
import pyttsx3
import threading

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize ASL detector and text processor
asl_detector = ASLDetector()
text_processor = TextProcessor()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Global variables
current_letter = None
letter_buffer = ""
processing_lock = False

def speak_text(text):
    """Speak the given text using text-to-speech engine"""
    engine.say(text)
    engine.runAndWait()

def generate_frames():
    """Generate frames from webcam with ASL detection"""
    global current_letter
    
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame to detect ASL
        processed_frame, detected_letter = asl_detector.process_frame(frame)
        
        # Update current letter
        if detected_letter:
            current_letter = detected_letter
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_letter')
def get_letter():
    """Return the currently detected letter"""
    global current_letter
    return jsonify({'letter': current_letter if current_letter else ''})

@app.route('/add_letter')
def add_letter():
    """Add current letter to buffer and speak it"""
    global letter_buffer, current_letter
    
    if current_letter:
        letter_buffer += current_letter
        
        # Create a thread to speak the letter to avoid blocking
        speech_thread = threading.Thread(target=speak_text, args=(current_letter,))
        speech_thread.start()
        
        return jsonify({
            'success': True, 
            'letter': current_letter,
            'buffer': letter_buffer
        })
    
    return jsonify({'success': False})

@app.route('/process_text')
def process_text():
    """Process buffered letters into words using Gemini AI"""
    global letter_buffer, processing_lock
    
    if not letter_buffer or processing_lock:
        return jsonify({'success': False, 'message': 'No text to process or processing in progress'})
    
    processing_lock = True
    
    try:
        # Process text to words
        processed_text = text_processor.process_text(letter_buffer)
        
        # Speak the processed text
        speech_thread = threading.Thread(target=speak_text, args=(processed_text,))
        speech_thread.start()
        
        # Reset buffer
        letter_buffer = ""
        
        return jsonify({
            'success': True,
            'processed_text': processed_text
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        processing_lock = False

@app.route('/clear_buffer')
def clear_buffer():
    """Clear the letter buffer"""
    global letter_buffer
    letter_buffer = ""
    return jsonify({'success': True})

@app.route('/get_buffer')
def get_buffer():   
    """Get the current letter buffer"""
    global letter_buffer
    return jsonify({'buffer': letter_buffer})

if __name__ == '__main__':
    app.run(debug=True)