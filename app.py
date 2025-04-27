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

hand_size_calibration = 1.0
is_calibrating = False
calibration_frames = []

def calibrate_hand_size(hand_landmarks):
    """
    Enhanced calibration for hand size by measuring multiple finger dimensions
    and palm width to get a more accurate hand size estimate.
    """
    global hand_size_calibration, calibration_frames, is_calibrating
    

    measurements = []
    
    wrist = hand_landmarks.landmark[0]
    for tip_idx in [4, 8, 12, 16, 20]:  # Thumb, Index, Middle, Ring, Pinky tips
        tip = hand_landmarks.landmark[tip_idx]
        distance = ((tip.x - wrist.x)**2 + 
                    (tip.y - wrist.y)**2 + 
                    (tip.z - wrist.z)**2)**0.5
        measurements.append(distance)
    
    index_mcp = hand_landmarks.landmark[5]
    pinky_mcp = hand_landmarks.landmark[17]
    palm_width = ((pinky_mcp.x - index_mcp.x)**2 + 
                  (pinky_mcp.y - index_mcp.y)**2 + 
                  (pinky_mcp.z - index_mcp.z)**2)**0.5
    measurements.append(palm_width)
    
    middle_mcp = hand_landmarks.landmark[9]
    middle_tip = hand_landmarks.landmark[12]
    middle_length = ((middle_tip.x - middle_mcp.x)**2 + 
                     (middle_tip.y - middle_mcp.y)**2 + 
                     (middle_tip.z - middle_mcp.z)**2)**0.5
    measurements.append(middle_length)
    
    frame_avg = sum(measurements) / len(measurements)
    calibration_frames.append(frame_avg)
    
    if len(calibration_frames) >= 30:
        sorted_frames = sorted(calibration_frames)
        q1_idx = len(sorted_frames) // 4
        q3_idx = q1_idx * 3
        q1 = sorted_frames[q1_idx]
        q3 = sorted_frames[q3_idx]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filtered_values = [v for v in calibration_frames 
                          if lower_bound <= v <= upper_bound]
        
        if filtered_values:
            hand_size_calibration = sum(filtered_values) / len(filtered_values)
            print(f"Enhanced hand size calibration complete: {hand_size_calibration:.4f}")
        
        is_calibrating = False
        calibration_frames = []
    
    return len(calibration_frames) / 30.0  # Return progress (0.0-1.0)
@app.route('/calibrate', methods=['POST'])
def start_calibration():
    """Start the hand size calibration process"""
    global is_calibrating, calibration_frames
    is_calibrating = True
    calibration_frames = []
    return jsonify({"status": "calibration_started"})


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

try:
    asl_model = tf.keras.models.load_model('models/letter_model.h5')
    model_loaded = True
except:
    model_loaded = False
    print("Warning: ASL model not found. App will run in demo mode.")

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

detected_letters = []
current_mode = "letter"  # Default mode
last_detection_time = time.time()
detection_cooldown = 1.0  # seconds between detections
speech_thread = None
last_spoken_text = ""

asl_classes = list(string.ascii_uppercase) + ["SPACE", "DELETE"]

def demo_predict_asl(hand_landmarks):
    """
    Use deterministic hand landmark-based detection instead of random guessing
    """
    detected_letter = detect_asl_by_hand_landmarks(hand_landmarks)
    
    return detected_letter

def predict_asl(frame, hand_landmarks):
    """
    Use the trained model to predict the ASL sign
    """
    if not model_loaded:
        return demo_predict_asl(hand_landmarks)
    
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    features = np.array(features).reshape(1, -1)
    
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
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
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
            
            visualize_finger_status(frame, hand_landmarks)
            
            # Only detect a new sign after cooldown period
            current_time = time.time()
            if current_time - last_detection_time >= detection_cooldown:
                if model_loaded:
                    predicted_letter = predict_asl(rgb_frame, hand_landmarks)
                else:
                    predicted_letter = detect_asl_by_hand_landmarks(hand_landmarks)
                
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
    
    for i, landmark in enumerate(hand_landmarks.landmark):
        x, y = int(landmark.x * w), int(landmark.y * h)
        
        # Fingertips (indices 4, 8, 12, 16, 20)
        if i in [4, 8, 12, 16, 20]:
            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
            
            finger_names = {4: "T", 8: "I", 12: "M", 16: "R", 20: "P"}
            cv2.putText(frame, finger_names.get(i, ""), (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        else:
            cv2.circle(frame, (x, y), 5, (0, 128, 255), -1)
    
    thumb_tip = hand_landmarks.landmark[4]
    index_base = hand_landmarks.landmark[5]
    
    thumb_tip_x, thumb_tip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
    index_base_x, index_base_y = int(index_base.x * w), int(index_base.y * h)
    
    distance = ((thumb_tip_x - index_base_x)**2 + (thumb_tip_y - index_base_y)**2)**0.5
    if distance < 50:  # Threshold in pixels
        cv2.line(frame, (thumb_tip_x, thumb_tip_y), (index_base_x, index_base_y), 
                (0, 255, 0), 2)
    
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    
    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
    middle_mcp_x, middle_mcp_y = int(middle_mcp.x * w), int(middle_mcp.y * h)
    
    dx = middle_mcp_x - wrist_x
    dy = middle_mcp_y - wrist_y
    
    length = 50
    if (dx**2 + dy**2)**0.5 > 0:
        dx = int(dx / (dx**2 + dy**2)**0.5 * length)
        dy = int(dy / (dx**2 + dy**2)**0.5 * length)
    
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
            processed_frame = process_frame(frame)
            
            text = ''.join(detected_letters[-20:])  # Show last 20 detected letters
            cv2.putText(processed_frame, text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            mode_text = f"Mode: {current_mode.capitalize()}"
            cv2.putText(processed_frame, mode_text, (10, frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def detect_asl_by_hand_landmarks(hand_landmarks):
    """
    Enhanced ASL letter detection based on hand landmarks
    Detects all 26 letters of the ASL alphabet plus SPACE and DELETE
    with improved accuracy using multiple detection methods
    """
    points = []
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
    for landmark in hand_landmarks.landmark:
        x, y, z = landmark.x, landmark.y, landmark.z
        points.append((x, y, z))
        min_x, max_x = min(min_x, x), max(max_x, x)
        min_y, max_y = min(min_y, y), max(max_y, y)
        min_z, max_z = min(min_z, z), max(max_z, z)
    
    width_x = max(0.001, max_x - min_x)  # Avoid division by zero
    width_y = max(0.001, max_y - min_y)
    width_z = max(0.001, max_z - min_z)
    
    norm_points = []
    for x, y, z in points:
        norm_x = (x - min_x) / width_x
        norm_y = (y - min_y) / width_y
        norm_z = (z - min_z) / width_z
        norm_points.append((norm_x, norm_y, norm_z))
    
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
    
    def distance(p1_idx, p2_idx, normalized=True):
        """Calculate Euclidean distance between two landmarks"""
        p_list = norm_points if normalized else points
        p1 = p_list[p1_idx]
        p2 = p_list[p2_idx]
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5
    
    def angle_between(p1_idx, p2_idx, p3_idx):
        """Calculate angle between three points with p2 as the vertex"""
        p1 = norm_points[p1_idx]
        p2 = norm_points[p2_idx]
        p3 = norm_points[p3_idx]
        
        v1 = (p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2])
        v2 = (p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2])
        
        dot_product = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        
        mag1 = (v1[0]**2 + v1[1]**2 + v1[2]**2)**0.5
        mag2 = (v2[0]**2 + v2[1]**2 + v2[2]**2)**0.5
        
        if mag1 * mag2 == 0:
            return 0
        
        cos_angle = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
        angle_rad = np.arccos(cos_angle)
        angle_deg = angle_rad * 180 / np.pi
        return angle_deg
    
    def is_finger_extended(tip_idx, pip_idx, mcp_idx, threshold=0.4):
        """Check if a finger is extended using both distance and angle"""
        tip_to_mcp_dist = distance(tip_idx, mcp_idx)
        if tip_to_mcp_dist < threshold:
            return False
        
        angle = angle_between(tip_idx, pip_idx, mcp_idx)
        return angle > 160
    
    def is_finger_bent(tip_idx, pip_idx, mcp_idx):
        """Check if a finger is bent (not straight)"""
        angle = angle_between(tip_idx, pip_idx, mcp_idx)
        return angle < 160
    
    def is_finger_curled(tip_idx, mcp_idx):
        """Check if a finger is curled toward palm"""
        tip_to_wrist = distance(tip_idx, WRIST)
        mcp_to_wrist = distance(mcp_idx, WRIST)
        
        return tip_to_wrist < mcp_to_wrist
    
    def is_thumb_extended():
        """Specific check for thumb extension"""
        angle = angle_between(THUMB_TIP, THUMB_IP, THUMB_MCP)
        
        tip_to_cmc_dist = distance(THUMB_TIP, THUMB_CMC)
        
        return angle > 150 and tip_to_cmc_dist > 0.3
    
    def are_fingers_touching(tip1_idx, tip2_idx, threshold=0.15):
        """Check if two fingertips are close to each other"""
        return distance(tip1_idx, tip2_idx) < threshold
    
    def is_palm_facing_camera():
        """Determine if palm is facing the camera based on z-coordinates"""
        palm_z = sum(points[i][2] for i in [0, 1, 5, 9, 13, 17]) / 6
        fingertips_z = sum(points[i][2] for i in [4, 8, 12, 16, 20]) / 5
        
        return fingertips_z < palm_z
    
    def hand_orientation():
        """Determine hand orientation (vertical/horizontal)"""
        wrist = norm_points[WRIST]
        middle_mcp = norm_points[MIDDLE_MCP]
        
        v_hand = (middle_mcp[0]-wrist[0], middle_mcp[1]-wrist[1], 0)
        v_vert = (0, 1, 0)
        
        dot_product = v_hand[1]  # dot product with (0,1,0) is just y component
        mag_hand = (v_hand[0]**2 + v_hand[1]**2)**0.5
        
        if mag_hand == 0:
            return "unknown"
        
        cos_angle = dot_product / mag_hand
        angle_deg = np.arccos(max(-1.0, min(1.0, cos_angle))) * 180 / np.pi
        
        # Classify orientation
        if angle_deg < 30:
            return "up"
        elif angle_deg > 150:
            return "down"
        else:
            # Check if pointing left or right
            if middle_mcp[0] > wrist[0]:
                return "right"
            else:
                return "left"
    
    thumb_extended = is_thumb_extended()
    index_extended = is_finger_extended(INDEX_TIP, INDEX_PIP, INDEX_MCP)
    middle_extended = is_finger_extended(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    ring_extended = is_finger_extended(RING_TIP, RING_PIP, RING_MCP)
    pinky_extended = is_finger_extended(PINKY_TIP, PINKY_PIP, PINKY_MCP)
    
    index_bent = is_finger_bent(INDEX_TIP, INDEX_PIP, INDEX_MCP) and not is_finger_curled(INDEX_TIP, INDEX_MCP)
    middle_bent = is_finger_bent(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP) and not is_finger_curled(MIDDLE_TIP, MIDDLE_MCP)
    ring_bent = is_finger_bent(RING_TIP, RING_PIP, RING_MCP) and not is_finger_curled(RING_TIP, RING_MCP)
    pinky_bent = is_finger_bent(PINKY_TIP, PINKY_PIP, PINKY_MCP) and not is_finger_curled(PINKY_TIP, PINKY_MCP)
    
    index_curled = is_finger_curled(INDEX_TIP, INDEX_MCP)
    middle_curled = is_finger_curled(MIDDLE_TIP, MIDDLE_MCP)
    ring_curled = is_finger_curled(RING_TIP, RING_MCP)
    pinky_curled = is_finger_curled(PINKY_TIP, PINKY_MCP)
    
    # Get additional hand configuration information
    palm_facing = is_palm_facing_camera()
    orientation = hand_orientation()
    
    # Additional spatial relationships between fingers
    thumb_index_touching = are_fingers_touching(THUMB_TIP, INDEX_TIP)
    thumb_middle_touching = are_fingers_touching(THUMB_TIP, MIDDLE_TIP)
    index_middle_touching = are_fingers_touching(INDEX_TIP, MIDDLE_TIP)
    middle_ring_touching = are_fingers_touching(MIDDLE_TIP, RING_TIP)
    ring_pinky_touching = are_fingers_touching(RING_TIP, PINKY_TIP)
    
    # Distance between fingertips - useful for many signs
    index_middle_dist = distance(INDEX_TIP, MIDDLE_TIP)
    middle_ring_dist = distance(MIDDLE_TIP, RING_TIP)
    ring_pinky_dist = distance(RING_TIP, PINKY_TIP)
    
    # Check if thumb crosses palm (important for many ASL letters)
    thumb_crosses_palm = (norm_points[THUMB_TIP][0] > norm_points[INDEX_MCP][0]) if palm_facing else False
    
    # Check thumb position relative to other fingers
    thumb_below_fingers = norm_points[THUMB_TIP][1] > norm_points[INDEX_MCP][1]
    
    # Special configurations
    thumb_between_index_middle = (
        distance(THUMB_TIP, INDEX_MCP) < 0.2 and
        distance(THUMB_TIP, MIDDLE_MCP) < 0.2
    )
    
    # ASL Letter detection logic with comprehensive rules
    
    # A - Fist with thumb at side
    if (not index_extended and not middle_extended and not ring_extended and 
        not pinky_extended and thumb_extended and not thumb_crosses_palm):
        return "A"
        
    # B - Fingers extended and together, thumb across palm
    elif (index_extended and middle_extended and ring_extended and pinky_extended and
          not thumb_extended and index_middle_dist < 0.15 and middle_ring_dist < 0.15 and
          ring_pinky_dist < 0.15 and palm_facing):
        return "B"
        
    # C - Curved hand shape, all fingers bent in same direction
    elif (not index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and not thumb_extended and
          index_bent and middle_bent and ring_bent and pinky_bent and 
          palm_facing and orientation != "down"):
        return "C"
        
    # D - Index extended, others curled, thumb touches middle finger
    elif (index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and distance(THUMB_TIP, MIDDLE_PIP) < 0.2):
        return "D"
        
    # E - Fingers curled in, thumb across fingers
    elif (index_curled and middle_curled and ring_curled and pinky_curled and
          thumb_crosses_palm and palm_facing):
        return "E"
        
    # F - Thumb and index touching, other fingers extended
    elif (thumb_index_touching and not index_extended and 
          middle_extended and ring_extended and pinky_extended):
        return "F"
        
    # G - Thumb and index pointing horizontally, index bent, others curled
    elif (thumb_extended and index_bent and not middle_extended and 
          not ring_extended and not pinky_extended and 
          orientation in ["right", "left"]):
        return "G"
        
    # H - Index and middle extended and parallel, others curled
    elif (index_extended and middle_extended and not ring_extended and 
          not pinky_extended and index_middle_dist < 0.15 and
          orientation in ["right", "left"]):
        return "H"
        
    # I - Pinky extended, others curled, palm facing side
    elif (not index_extended and not middle_extended and not ring_extended and 
          pinky_extended and not palm_facing):
        return "I"
        
    # J - Like I but with motion (simplified as pinky extended with palm facing side and curved)
    elif (not index_extended and not middle_extended and not ring_extended and 
          pinky_bent and not palm_facing and distance(PINKY_TIP, WRIST) > 0.35):
        return "J"
        
    # K - Index and middle extended in V shape, palm facing side
    elif (index_extended and middle_extended and not ring_extended and 
          not pinky_extended and index_middle_dist > 0.25 and
          not palm_facing and orientation in ["up", "left", "right"]):
        return "K"
        
    # L - L shape with thumb and index extended at 90°
    elif (index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_extended and distance(THUMB_TIP, INDEX_TIP) > 0.35 and
          orientation in ["up", "right"]):
        return "L"
        
    # M - Thumb between ring and pinky, fingers folded
    elif (not index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and index_curled and middle_curled and ring_curled and 
          distance(THUMB_TIP, RING_MCP) < 0.15):
        return "M"
        
    # N - Thumb between middle and ring, fingers folded
    elif (not index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and index_curled and middle_curled and 
          distance(THUMB_TIP, MIDDLE_MCP) < 0.15):
        return "N"
        
    # O - Fingertips form circle with thumb
    elif (not index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_index_touching and 
          index_bent and middle_bent and ring_bent and pinky_bent):
        return "O"
        
    # P - Thumb between index and middle, index pointing down
    elif (index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_extended and
          thumb_between_index_middle and orientation == "down"):
        return "P"
        
    # Q - Thumb and index pointing down
    elif (index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_extended and orientation == "down"):
        return "Q"
        
    # R - Index and middle crossed
    elif (index_extended and middle_extended and not ring_extended and 
          not pinky_extended and index_middle_touching and
          norm_points[INDEX_TIP][0] < norm_points[MIDDLE_TIP][0]):
        return "R"
        
    # S - Fist with thumb across fingers
    elif (index_curled and middle_curled and ring_curled and pinky_curled and
          thumb_crosses_palm and not thumb_extended):
        return "S"
        
    # T - Thumb between index and middle (but not pointing down like P)
    elif (not index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_extended and
          thumb_between_index_middle and orientation != "down"):
        return "T"
        
    # U - Index and middle extended parallel
    elif (index_extended and middle_extended and not ring_extended and 
          not pinky_extended and index_middle_dist < 0.2 and
          orientation == "up"):
        return "U"
        
    # V - Index and middle extended in V shape
    elif (index_extended and middle_extended and not ring_extended and 
          not pinky_extended and index_middle_dist > 0.25 and
          norm_points[INDEX_TIP][1] < norm_points[INDEX_PIP][1] and
          norm_points[MIDDLE_TIP][1] < norm_points[MIDDLE_PIP][1]):
        return "V"
        
    # W - Index, middle and ring extended in spread pattern
    elif (index_extended and middle_extended and ring_extended and 
          not pinky_extended and 
          index_middle_dist > 0.2 and middle_ring_dist > 0.2):
        return "W"
        
    # X - Index bent with thumb at side
    elif (not index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and thumb_extended and
          index_bent and not index_extended):
        return "X"
        
    # Y - Thumb and pinky extended only
    elif (not index_extended and not middle_extended and not ring_extended and 
          pinky_extended and thumb_extended and
          distance(THUMB_TIP, PINKY_TIP) > 0.4):
        return "Y"
        
    # Z - Index pointing forward with palm down (simplified as index extended horizontally)
    elif (index_extended and not middle_extended and not ring_extended and 
          not pinky_extended and orientation in ["left", "right"] and
          not palm_facing):
        return "Z"
    
    # SPACE - Open palm with fingers spread
    elif (index_extended and middle_extended and ring_extended and pinky_extended and
          thumb_extended and index_middle_dist > 0.15 and middle_ring_dist > 0.15 and
          ring_pinky_dist > 0.15 and palm_facing):
        return "SPACE"
        
    # DELETE - Closed fist with thumb pointing down
    elif (not index_extended and not middle_extended and not ring_extended and
          not pinky_extended and thumb_extended and
          orientation == "down" and thumb_below_fingers):
        return "DELETE"
    
    # Confidence check - if we are here, we didn't match any letter strongly
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