# SigniFi

## Overview

SigniFi is a real-time American Sign Language translation application that uses computer vision and machine learning to recognize and interpret ASL hand gestures. The application captures video from a webcam, processes hand movements using MediaPipe, identifies ASL letters and signs, and converts them into text and speech.

## Features

- **Real-time ASL recognition**: Detect and translate ASL alphabet signs instantly
- **Two recognition modes**: Letter mode and Word mode
- **Text-to-Speech**: Hear the interpreted signs spoken aloud
- **Hand pose visualization**: Visual feedback for better signing understanding
- **Customizable settings**: Adjust detection sensitivity and preferences
- **Hand size calibration**: Adapts to different hand sizes for improved accuracy
- **Detection history**: View recent detections for better tracking
- **Responsive UI**: Modern, user-friendly interface that works on different devices

## Technology Stack

### Backend
- **Python 3.8+**: Core programming language
- **Flask**: Web framework for serving the application
- **OpenCV**: Computer vision library for image processing
- **MediaPipe**: Google's ML solution for hand tracking and landmark detection
- **TensorFlow**: Machine learning framework for ASL sign classification
- **Google Generative AI (Gemini)**: For segmenting letter streams into words
- **gTTS (Google Text-to-Speech)**: For converting text to speech

### Frontend
- **HTML/CSS/JavaScript**: Core web technologies
- **Bootstrap 5**: For responsive design and UI components
- **Font Awesome 6**: For icons

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Internet connection (for Gemini API and TTS)

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/asl-interpreter.git
cd asl-interpreter
```

### Step 2: Create and activate a virtual environment
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

If you don't have a requirements.txt file, install these packages:
```bash
pip install flask opencv-python mediapipe tensorflow google-generativeai gtts numpy
```

### Step 4: Set up your Google API key (for Gemini)
Replace the placeholder API key in app.py with your own:
```python
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
```

### Step 5: Run the application
```bash
python app.py
```

The application will start a local server, typically at http://127.0.0.1:5000/

## Usage

1. Open your web browser and navigate to http://127.0.0.1:5000/
2. Allow camera permissions when prompted
3. Position your hand in the camera view
4. Make ASL letter signs with your dominant hand
5. The application will display the recognized letters and words
6. Use the control buttons to switch modes, clear text, or have the text spoken

### Recognition Modes

- **Letter Mode**: Recognizes individual ASL letters and displays them in sequence
- **Word Mode**: Attempts to group recognized letters into words using AI-powered segmentation

### Controls

- **Mode Selection**: Toggle between Letter and Word modes
- **Clear**: Reset the accumulated text
- **Copy**: Copy the interpreted text to clipboard
- **Speak**: Use text-to-speech to hear the interpretation
- **Settings**: Customize detection cooldown and other preferences

## Hand Size Calibration

For accurate detection across different hand sizes, the application includes a calibration feature:

1. Click the "Settings" gear icon
2. Select "Calibrate Hand Size"
3. Follow the prompts to hold your hand in the frame
4. The system will calibrate to your hand size after a few seconds

## Technical Details

### Hand Detection and Tracking

The application uses MediaPipe Hands to detect and track hands in the webcam feed. MediaPipe provides 21 3D landmarks for each hand, which are used for gesture recognition.

### ASL Recognition

Two methods of ASL recognition are implemented:

1. **Rule-based detection**: Uses geometric relationships between hand landmarks to identify letters (for demo mode)
2. **Machine learning model**: A TensorFlow model trained on ASL hand gestures (when available)

### Word Segmentation

In Word mode, the application uses Google's Gemini AI to intelligently segment streams of letters into meaningful words and phrases.

## Development Challenges

### ASL Detection Accuracy

Detecting ASL signs accurately presented several challenges:

- **Similar hand poses**: Many ASL letters have similar hand poses (like E and A, or M and N)
- **Hand orientation**: Detection quality varies with different hand orientations
- **Dynamic signs**: Some ASL signs involve movement (like J and Z) which are harder to detect from static frames
- **Background interference**: Noisy backgrounds can affect detection quality

Solutions implemented:
- Hand landmark relationships rather than raw pixel data
- Detection cooldown to prevent rapid fluctuations
- Visualization of finger states to help users adjust their signs

### OpenCV Integration

Working with OpenCV presented some technical hurdles:

- **Frame processing speed**: Balancing processing intensity with real-time performance
- **Image quality**: Dealing with poor lighting or camera quality
- **Memory management**: Efficient processing of continuous video streams

### Webcam Calibration

Each user has different hand sizes and webcam setups, requiring:

- **Hand size normalization**: Implementing calibration to account for different hand sizes
- **Distance adaptation**: Adjusting detection sensitivity based on distance from camera
- **Lighting conditions**: Providing guidance for optimal lighting setups

### Learning ASL

Implementing proper ASL detection required:

- **ASL alphabet research**: Understanding the nuances of proper hand positioning
- **Consulting ASL resources**: Working with reference materials to validate signs
- **Edge cases**: Accounting for natural variations in how people form signs

## Future Improvements

- Full word and phrase recognition beyond just the alphabet
- Fingerspelling detection improvements
- Support for ASL grammar and syntax
- Mobile app version
- Two-way translation (text to ASL visualization)
- Offline mode that doesn't require internet connection
- Support for other sign languages

## FAQ

### How accurate is the ASL detection?

The accuracy varies depending on several factors:
- **Lighting**: Good lighting significantly improves detection
- **Hand positioning**: Clear view of your hand with minimal background clutter helps
- **Speed**: Holding signs steady for a moment improves recognition
- **Calibration**: Proper hand size calibration improves accuracy for your specific hand

The system works best with the standard ASL alphabet signs in good lighting conditions.

### Why does the system sometimes detect the wrong letter?

ASL signs that are similar in appearance may be confused. For example:
- M and N signs are very similar
- E and A both involve a closed fist with slight differences
- R and U can look similar from certain angles
- Angles matter for detection
- The model is rushed, and requires far more training than two days

Practice holding signs clearly and adjust your hand position if a sign is consistently misinterpreted.

### Can it detect ASL beyond the alphabet?

The current version focuses on the ASL alphabet, numbers, and a few special signs (SPACE, DELETE). Full word detection using native ASL signs is planned for future versions.

### Do I need an internet connection?

Yes, the application uses:
- Google's Gemini API for word segmentation
- Google Text-to-Speech for audio output

A future offline-only mode is planned.

### How can I improve detection quality?

- Ensure good lighting
- Position your hand against a plain background
- Hold signs steady for a moment
- Complete the hand size calibration
- Maintain a consistent distance from the camera
- Make sure your whole hand is visible in the frame

### Why doesn't the application recognize my signs immediately?

There's an intentional cooldown period between detections to prevent accidental inputs. This can be adjusted in the settings.

### How can I train the system on my own signing style?

The current version doesn't include personalized training. However, the hand size calibration helps adapt to your specific hand proportions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe team for their hand tracking solution
- TensorFlow for the machine learning framework
- Flask for the web framework
- Google Generative AI team for Gemini API
- The ASL community for resources and guidance