import cv2
import numpy as np
import time

class ASLDetector:
    def __init__(self):
        """Initialize ASL detector with necessary parameters and model"""
        # ASL alphabet letters (excluding J and Z which require motion)
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                        'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        
        # Region of interest parameters
        self.roi_top = 100
        self.roi_bottom = 400
        self.roi_left = 100
        self.roi_right = 400
        
        # Detection parameters
        self.min_confidence = 0.7
        self.last_detection = None
        self.detection_cooldown = 1.0  # seconds
        self.last_detection_time = 0
        
        # For demonstration, we're using a mock detector
        # In a real application, you would load a trained model here

    def process_frame(self, frame):
        """Process a frame to detect ASL letters"""
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Draw the ROI (region of interest)
        cv2.rectangle(output_frame, 
                      (self.roi_left, self.roi_top), 
                      (self.roi_right, self.roi_bottom), 
                      (0, 255, 0), 2)
        
        # Extract the ROI for processing
        roi = frame[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Apply thresholding to segment hand
        _, thresholded = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_letter = None
        
        # If contours are found, process them
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            
            # Only process if the contour is large enough
            if cv2.contourArea(max_contour) > 1000:
                # Draw contour on the output frame
                cv2.drawContours(output_frame[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right],
                                [max_contour], 0, (0, 0, 255), 2)
                
                # Create a convex hull around the contour
                hull = cv2.convexHull(max_contour)
                cv2.drawContours(output_frame[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right],
                                [hull], 0, (255, 0, 0), 2)
                
                # In a real application, you would extract features here and use a trained model
                # For now, we'll mock a letter detection based on the current time
                
                current_time = time.time()
                if current_time - self.last_detection_time > self.detection_cooldown:
                    # Implement your actual detection logic here - for now we'll use a simple mock
                    # This would be replaced with actual image recognition in a real application 
                    # For demonstration only - pretend we detected a letter:
                    features = self._extract_features(max_contour, hull, thresholded)
                    detected_letter = self._classify_hand_shape(features)
                    
                    if detected_letter:
                        self.last_detection = detected_letter
                        self.last_detection_time = current_time
        
        # Display the current detection on the frame
        if self.last_detection:
            cv2.putText(output_frame, f"Detected: {self.last_detection}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        return output_frame, self.last_detection
    
    def _extract_features(self, contour, hull, thresholded):
        """Extract features from the hand contour"""
        # Features dictionary
        features = {}
        
        # Area of contour and hull
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)
        
        # Solidity (ratio of contour area to hull area)
        features['solidity'] = float(contour_area) / hull_area if hull_area > 0 else 0
        
        # Aspect ratio of the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        features['aspect_ratio'] = float(w) / h if h > 0 else 0
        
        # Find convexity defects
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        try:
            defects = cv2.convexityDefects(contour, hull_indices)
            features['defect_count'] = len(defects) if defects is not None else 0
        except:
            features['defect_count'] = 0
        
        # Count fingers using the convexity defects
        features['finger_count'] = self._count_fingers(contour, defects) if 'defects' in locals() and defects is not None else 0
        
        return features
    
    def _count_fingers(self, contour, defects):
        """Count fingers using convexity defects"""
        finger_count = 0
        
        if defects is None:
            return finger_count
        
        for i in range(defects.shape[0]):
            start_idx, end_idx, farthest_idx, _ = defects[i, 0]
            start = tuple(contour[start_idx][0])
            end = tuple(contour[end_idx][0])
            far = tuple(contour[farthest_idx][0])
            
            # Calculate the angle between fingers
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            
            # If the angle is less than 90 degrees, it's likely a finger
            if angle <= np.pi / 2:
                finger_count += 1
        
        # Add 1 for the last finger
        return finger_count + 1
    
    def _classify_hand_shape(self, features):
        """Classify hand shape into ASL letter based on features"""
        # This is a simplified mock classification for demonstration
        # In a real app, you would use a trained model (CNN, SVM, etc.)
        
        # Randomly select a letter for demo purposes - replace with actual classification
        import random
        import time
        
        # Use the current second to "randomly" select a letter but with some consistency
        second = int(time.time()) % 5
        if second == 0:
            return 'A'
        elif second == 1:
            return 'B'
        elif second == 2:
            return 'C'
        elif second == 3:
            return 'D'
        elif second == 4:
            return 'E'
        
        # In a real application, you'd use something like:
        # return self.model.predict(features)