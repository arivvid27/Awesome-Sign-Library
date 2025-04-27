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

# Add these variables at the top of the file
hand_size_calibration = 1.0  # Default calibration value
is_calibrating = False
calibration_frames = []

def calibrate_hand_size(hand_landmarks):
    """
    Calibrate for hand size by measuring average finger lengths
    """
    global hand_size_calibration, calibration_frames, is_calibrating
    
    # Extract key points
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    middle_tip = hand_landmarks.landmark[12]
    
    # Measure hand size (distance from wrist to middle finger tip)
    wrist_to_middle_tip = ((middle_tip.x - wrist.x)**2 + 
                           (middle_tip.y - wrist.y)**2 + 
                           (middle_tip.z - wrist.z)**2)**0.5
    
    # Add to calibration frames
    calibration_frames.append(wrist_to_middle_tip)
    
    # If we have enough frames, compute the average
    if len(calibration_frames) >= 30:
        # Remove outliers (values too far from median)
        median_value = np.median(calibration_frames)
        filtered_values = [v for v in calibration_frames 
                          if abs(v - median_value) < 0.1]
        
        if filtered_values:
            # Set calibration value based on average
            hand_size_calibration = sum(filtered_values) / len(filtered_values)
            print(f"Hand size calibration complete: {hand_size_calibration:.4f}")
        
        # Reset calibration state
        is_calibrating = False
        calibration_frames = []
    
    return len(calibration_frames) / 30.0  # Return progress (0.0-1.0)

# Add a new endpoint for the calibration
@app.route('/calibrate', methods=['POST'])
def start_calibration():
    """Start the hand size calibration process"""
    global is_calibrating, calibration_frames
    is_calibrating = True
    calibration_frames = []
    return jsonify({"status": "calibration_started"})


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
    GOOGLE_API_KEY = "AIzaSyCfb1AlLLYl9V3gEODD1JKwsuLTqQi0E3Q"
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
    Use deterministic hand landmark-based detection instead of random guessing
    """
    # Use our landmark-based detection instead of random selection
    detected_letter = detect_asl_by_hand_landmarks(hand_landmarks)
    
    # Only return the letter if we're confident in the detection
    # Otherwise return None (no detection)
    return detected_letter

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
    
    # Draw detection info on the frame
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, h-50), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "Current text: " + ''.join(detected_letters[-20:]), 
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Add finger status visualization
            visualize_finger_status(frame, hand_landmarks)
            
            # Only detect a new sign after cooldown period
            current_time = time.time()
            if current_time - last_detection_time >= detection_cooldown:
                # Try to detect the ASL sign
                if model_loaded:
                    predicted_letter = predict_asl(rgb_frame, hand_landmarks)
                else:
                    predicted_letter = detect_asl_by_hand_landmarks(hand_landmarks)
                
                # Display detection attempt
                if predicted_letter is None:
                    cv2.putText(frame, "Detecting...", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    # Process the detected letter
                    cv2.putText(frame, f"Detected: {predicted_letter}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
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
                        if predicted_letter == "SPACE":
                            text = ''.join(detected_letters[-10:])  # Last few letters
                            speak_in_background(text.strip())
                    
                    # Update detection time
                    last_detection_time = current_time
    else:
        cv2.putText(frame, "No hand detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame
def visualize_finger_status(frame, hand_landmarks):
    """
    Visualize finger positions and states to help with hand pose understanding
    """
    h, w, _ = frame.shape
    
    # Draw fingertip markers
    for i, landmark in enumerate(hand_landmarks.landmark):
        x, y = int(landmark.x * w), int(landmark.y * h)
        
        # Fingertips (indices 4, 8, 12, 16, 20)
        if i in [4, 8, 12, 16, 20]:
            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
            
            # Label the fingertips
            finger_names = {4: "T", 8: "I", 12: "M", 16: "R", 20: "P"}
            cv2.putText(frame, finger_names.get(i, ""), (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Other joints
        else:
            cv2.circle(frame, (x, y), 5, (0, 128, 255), -1)
    
    # Draw connecting lines between specific landmarks for better visualization
    # Connect thumb tip to index base (useful for some letter signs)
    thumb_tip = hand_landmarks.landmark[4]
    index_base = hand_landmarks.landmark[5]
    
    thumb_tip_x, thumb_tip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
    index_base_x, index_base_y = int(index_base.x * w), int(index_base.y * h)
    
    # Check if thumb is close to index base (which is common in many ASL signs)
    distance = ((thumb_tip_x - index_base_x)**2 + (thumb_tip_y - index_base_y)**2)**0.5
    if distance < 50:  # Threshold in pixels
        cv2.line(frame, (thumb_tip_x, thumb_tip_y), (index_base_x, index_base_y), 
                (0, 255, 0), 2)
    
    # Draw a hand orientation indicator
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    
    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
    middle_mcp_x, middle_mcp_y = int(middle_mcp.x * w), int(middle_mcp.y * h)
    
    # Hand direction vector
    dx = middle_mcp_x - wrist_x
    dy = middle_mcp_y - wrist_y
    
    # Normalize and scale for arrow
    length = 50
    if (dx**2 + dy**2)**0.5 > 0:
        dx = int(dx / (dx**2 + dy**2)**0.5 * length)
        dy = int(dy / (dx**2 + dy**2)**0.5 * length)
    
    # Draw arrow for hand orientation
    cv2.arrowedLine(frame, (wrist_x, wrist_y), (wrist_x + dx, wrist_y + dy), 
                   (255, 0, 0), 2)
    
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

# Add this function to app.py after the existing functions

def detect_asl_by_hand_landmarks(hand_landmarks):
    """
    Comprehensive ASL letter detection based on hand landmarks
    Detects all 26 letters of the ASL alphabet plus SPACE and DELETE
    """
    # Extract coordinates of all hand landmarks
    points = []
    for landmark in hand_landmarks.landmark:
        points.append((landmark.x, landmark.y, landmark.z))
    
    # Key landmark indices
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    # Helper functions
    def distance(p1_idx, p2_idx):
        """Calculate Euclidean distance between two landmarks"""
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5
    
    def is_finger_extended(tip_idx, pip_idx, mcp_idx, threshold=0.09):
        """Check if a finger is extended (tip is far from base)"""
        # Measure ratio of distances to account for different hand sizes
        tip_to_pip = distance(tip_idx, pip_idx)
        pip_to_mcp = distance(pip_idx, mcp_idx)
        mcp_to_wrist = distance(mcp_idx, WRIST)
        
        # A finger is extended if the tip is far from the PIP joint
        # relative to the distance from PIP to MCP
        return tip_to_pip > threshold * mcp_to_wrist
    
    def is_thumb_extended(threshold=0.08):
        """Check if thumb is extended (special case)"""
        thumb_tip_to_ip = distance(THUMB_TIP, THUMB_IP)
        ip_to_mcp = distance(THUMB_IP, THUMB_MCP)
        mcp_to_cmc = distance(THUMB_MCP, THUMB_CMC)
        
        return thumb_tip_to_ip > threshold * (ip_to_mcp + mcp_to_cmc)
    
    def is_finger_curled(tip_idx, mcp_idx, threshold=0.1):
        """Check if a finger is curled inward"""
        tip = points[tip_idx]
        mcp = points[mcp_idx]
        wrist = points[WRIST]
        
        # Check if the tip is closer to the wrist than the MCP joint
        tip_to_wrist = ((tip[0]-wrist[0])**2 + (tip[1]-wrist[1])**2)**0.5
        mcp_to_wrist = ((mcp[0]-wrist[0])**2 + (mcp[1]-wrist[1])**2)**0.5
        
        return tip_to_wrist < mcp_to_wrist - threshold
    
    def is_finger_bent(tip_idx, pip_idx, mcp_idx, threshold=0.05):
        """Check if a finger is bent (not straight)"""
        tip = points[tip_idx]
        pip = points[pip_idx]
        mcp = points[mcp_idx]
        
        # Calculate the angle between the joints
        vec1 = [pip[0]-mcp[0], pip[1]-mcp[1]]
        vec2 = [tip[0]-pip[0], tip[1]-pip[1]]
        
        # Normalize vectors
        vec1_norm = (vec1[0]**2 + vec1[1]**2)**0.5
        vec2_norm = (vec2[0]**2 + vec2[1]**2)**0.5
        
        if vec1_norm == 0 or vec2_norm == 0:
            return False
        
        # Calculate dot product
        dot_product = (vec1[0]*vec2[0] + vec1[1]*vec2[1]) / (vec1_norm * vec2_norm)
        dot_product = max(-1, min(1, dot_product))  # Ensure within valid range
        
        # Convert to angle
        angle = np.arccos(dot_product) * 180 / np.pi
        
        return angle > threshold
    
    def are_fingers_touching(tip1_idx, tip2_idx, threshold=0.05):
        """Check if two fingertips are touching"""
        return distance(tip1_idx, tip2_idx) < threshold
    
    def fingertip_y_position(tip_idx, reference_idx):
        """Get relative Y position of fingertip compared to reference point"""
        return points[tip_idx][1] - points[reference_idx][1]
    
    def fingertip_x_position(tip_idx, reference_idx):
        """Get relative X position of fingertip compared to reference point"""
        return points[tip_idx][0] - points[reference_idx][0]
    
    # Check finger states
    thumb_extended = is_thumb_extended()
    index_extended = is_finger_extended(INDEX_TIP, INDEX_PIP, INDEX_MCP)
    middle_extended = is_finger_extended(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    ring_extended = is_finger_extended(RING_TIP, RING_PIP, RING_MCP)
    pinky_extended = is_finger_extended(PINKY_TIP, PINKY_PIP, PINKY_MCP)
    
    # Additional states
    index_curled = is_finger_curled(INDEX_TIP, INDEX_MCP)
    middle_curled = is_finger_curled(MIDDLE_TIP, MIDDLE_MCP)
    ring_curled = is_finger_curled(RING_TIP, RING_MCP)
    pinky_curled = is_finger_curled(PINKY_TIP, PINKY_MCP)
    
    index_bent = is_finger_bent(INDEX_TIP, INDEX_PIP, INDEX_MCP)
    middle_bent = is_finger_bent(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    ring_bent = is_finger_bent(RING_TIP, RING_PIP, RING_MCP)
    pinky_bent = is_finger_bent(PINKY_TIP, PINKY_PIP, PINKY_MCP)
    
    # Check fingertip positions
    index_y = fingertip_y_position(INDEX_TIP, WRIST)
    middle_y = fingertip_y_position(MIDDLE_TIP, WRIST)
    ring_y = fingertip_y_position(RING_TIP, WRIST)
    pinky_y = fingertip_y_position(PINKY_TIP, WRIST)
    thumb_y = fingertip_y_position(THUMB_TIP, WRIST)
    
    # Check fingertip relationships
    thumb_to_index = distance(THUMB_TIP, INDEX_TIP)
    thumb_to_middle = distance(THUMB_TIP, MIDDLE_TIP)
    index_to_middle = distance(INDEX_TIP, MIDDLE_TIP)
    middle_to_ring = distance(MIDDLE_TIP, RING_TIP)
    ring_to_pinky = distance(RING_TIP, PINKY_TIP)
    
    # ASL Letter detection rules
    
    # A - Fist with thumb at side
    if (not index_extended and not middle_extended and not ring_extended and 
        not pinky_extended and thumb_extended):
        return "A"
        
    # B - Fingers extended and together, thumb across palm
    elif (index_extended and middle_extended and ring_extended and pinky_extended and
          not thumb_extended and index_to_middle < 0.05 and middle_to_ring < 0.05 and
          ring_to_pinky < 0.05):
        return "B"
        
    # C - Curved hand, fingers together
    elif (not index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and not thumb_extended and
          index_bent and middle_bent and ring_bent and pinky_bent and
          thumb_to_index < 0.15):
        return "C"
        
    # D - Index up, others curved, thumb touches middle
    elif (index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_to_middle < 0.08):
        return "D"
        
    # E - Fingers curled in, thumb across fingers
    elif (not index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and not thumb_extended and
          index_curled and middle_curled and ring_curled and pinky_curled):
        return "E"
        
    # F - Thumb and index touching, other fingers extended
    elif (are_fingers_touching(THUMB_TIP, INDEX_TIP) and
          middle_extended and ring_extended and pinky_extended):
        return "F"
        
    # G - Thumb and index pointing out horizontally, others curled
    elif (index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_extended and
          abs(index_y) < 0.05 and abs(thumb_y) < 0.05):
        return "G"
        
    # H - Index and middle extended together, others curled
    elif (index_extended and middle_extended and not ring_extended and 
          not pinky_extended and index_to_middle < 0.05):
        return "H"
        
    # I - Pinky extended, others curled
    elif (not index_extended and not middle_extended and not ring_extended and 
          pinky_extended):
        return "I"
        
    # J - Like I but with motion (simplified as pinky extended and bent)
    elif (not index_extended and not middle_extended and not ring_extended and 
          pinky_extended and pinky_bent):
        return "J"
        
    # K - Index and middle extended in V shape, thumb touches middle joint
    elif (index_extended and middle_extended and not ring_extended and 
          not pinky_extended and index_to_middle > 0.1):
        return "K"
        
    # L - L shape with thumb and index
    elif (index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_extended and
          thumb_to_index > 0.15 and 
          abs(fingertip_y_position(INDEX_TIP, INDEX_MCP)) < -0.1):
        return "L"
        
    # M - Thumb between ring and pinky, fingers folded
    elif (not index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_extended and
          are_fingers_touching(THUMB_TIP, RING_MCP)):
        return "M"
        
    # N - Thumb between middle and ring, fingers folded
    elif (not index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_extended and
          are_fingers_touching(THUMB_TIP, MIDDLE_MCP)):
        return "N"
        
    # O - Fingertips form circle with thumb
    elif (not index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and
          are_fingers_touching(THUMB_TIP, INDEX_TIP)):
        return "O"
        
    # P - Thumb between index and middle, index pointing
    elif (index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_extended and
          fingertip_x_position(THUMB_TIP, INDEX_MCP) < 0):
        return "P"
        
    # Q - Thumb and index pointing down
    elif (index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_extended and
          index_y > 0.1 and thumb_y > 0.1):
        return "Q"
        
    # R - Index and middle crossed
    elif (index_extended and middle_extended and not ring_extended and 
          not pinky_extended and
          index_to_middle < 0.05 and
          fingertip_x_position(INDEX_TIP, MIDDLE_TIP) < -0.03):
        return "R"
        
    # S - Fist with thumb in front of fingers
    elif (not index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_extended and
          fingertip_x_position(THUMB_TIP, INDEX_PIP) < -0.05):
        return "S"
        
    # T - Thumb between index and middle
    elif (not index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_extended and
          are_fingers_touching(THUMB_TIP, INDEX_MCP)):
        return "T"
        
    # U - Index and middle extended together
    elif (index_extended and middle_extended and not ring_extended and 
          not pinky_extended and index_to_middle < 0.05):
        return "U"
        
    # V - Index and middle extended in V shape
    elif (index_extended and middle_extended and not ring_extended and 
          not pinky_extended and index_to_middle > 0.1):
        return "V"
        
    # W - Index, middle and ring extended in spread pattern
    elif (index_extended and middle_extended and ring_extended and 
          not pinky_extended and 
          index_to_middle > 0.05 and middle_to_ring > 0.05):
        return "W"
        
    # X - Index bent with thumb at side
    elif (not index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_extended and
          index_bent):
        return "X"
        
    # Y - Thumb and pinky extended only
    elif (not index_extended and not middle_extended and not ring_extended and 
          pinky_extended and thumb_extended):
        return "Y"
        
    # Z - Index tracing Z shape (simplified as index extended pointing horizontally)
    elif (index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and
          abs(index_y) < 0.05):
        return "Z"
    
    # SPACE - Open palm with fingers spread
    elif (index_extended and middle_extended and ring_extended and pinky_extended and
          thumb_extended and index_to_middle > 0.07 and middle_to_ring > 0.07 and
          ring_to_pinky > 0.07):
        return "SPACE"
        
    # DELETE - Closed fist with thumb pointing down
    elif (not index_extended and not middle_extended and not ring_extended and
          not pinky_extended and thumb_extended and
          thumb_y > 0.1):
        return "DELETE"
    
    # Return None if no confident match
    return None

def fully_extended(tip, base, threshold=0.15):
    """Check if a finger is fully extended"""
    return distance(tip, base) > threshold

def partially_extended(tip, base, min_threshold=0.05, max_threshold=0.1):
    """Check if a finger is partially extended (useful for curved hand shapes)"""
    dist = distance(tip, base)
    return min_threshold < dist < max_threshold

def all_fingertips_close_together(points, threshold=0.1):
    """Check if all fingertips are close to each other (for O shape)"""
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if distance(points[i], points[j]) > threshold:
                return False
    return True

def distance(p1, p2):
    """Calculate Euclidean distance between two 3D points"""
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5

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