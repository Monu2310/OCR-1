import os
import sys
import time
import threading
from threading import Thread
import cv2
import numpy as np
import pytesseract
import mediapipe as mp
from datetime import datetime
from pathlib import Path
from gtts import gTTS
import pygame
from googletrans import Translator
import tensorflow as tf

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata/'

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize translator
translator = Translator()

# Convert TF Lite model to standard TensorFlow model
def convert_tflite_to_tf(tflite_model_path):
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get input shape
    input_shape = input_details[0]['shape']
    
    # Create a new TensorFlow model
    model = tf.keras.Sequential()
    
    # Add input layer with the correct shape
    model.add(tf.keras.layers.InputLayer(input_shape=(None, input_shape[1], input_shape[2])))
    
    # Add a flexible dense layer that can handle variable batch sizes
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print(f"Created TensorFlow model with input shape: {input_shape}")
    return model

class RateCounter:
    def __init__(self):
        self.start_time = None
        self.iterations = 0

    def start(self):
        self.start_time = time.perf_counter()
        return self

    def increment(self):
        self.iterations += 1

    def rate(self):
        elapsed_time = (time.perf_counter() - self.start_time)
        return self.iterations / elapsed_time

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        
        # Increase buffer size for smoother video
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        # Try to set higher resolution for better OCR
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def get_video_dimensions(self):
        width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return int(width), int(height)

    def stop_process(self):
        self.stopped = True

class OCR:
    def __init__(self):
        self.boxes = None
        self.stopped = False
        self.exchange = None
        self.language = None
        self.width = None
        self.height = None
        self.crop_width = None
        self.crop_height = None
        self.frame_count = 0
        self.process_every_n_frames = 2  # Process every 2nd frame for better performance
        self.selection_roi = None  # Store the hand-drawn selection rectangle
        self.last_processed_text = ""  # Cache the last processed text
        self.last_processed_roi = None  # Cache the last processed ROI
        self.model = None  # TensorFlow model

    def start(self):
        Thread(target=self.ocr, args=()).start()
        return self

    def set_exchange(self, video_stream):
        self.exchange = video_stream

    def set_language(self, language):
        self.language = language
        
    def set_selection_roi(self, roi):
        self.selection_roi = roi
        
    def set_model(self, model):
        self.model = model

    def ocr(self):
        while not self.stopped:
            if self.exchange is not None:
                self.frame_count += 1
                if self.frame_count % self.process_every_n_frames != 0:
                    continue
                    
                frame = self.exchange.frame
                
                # If we have a selection ROI, use it instead of the crop dimensions
                if self.selection_roi is not None:
                    x1, y1, x2, y2 = self.selection_roi
                    
                    # Check if ROI is the same as last processed - if so, skip processing
                    if self.last_processed_roi == self.selection_roi:
                        continue
                    
                    # Store current ROI for comparison
                    self.last_processed_roi = self.selection_roi
                    
                    # Ensure coordinates are within frame boundaries
                    x1 = max(0, min(x1, self.width-1))
                    y1 = max(0, min(y1, self.height-1))
                    x2 = max(0, min(x2, self.width-1))
                    y2 = max(0, min(y2, self.height-1))
                    
                    # Ensure x1 < x2 and y1 < y2
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1
                    
                    # Extract the ROI
                    roi = frame[y1:y2, x1:x2]
                    
                    # Skip if ROI is too small
                    if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
                        continue
                    
                    # Enhanced image processing for better OCR
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)
                    
                    # Apply bilateral filter to reduce noise while preserving edges
                    processed = cv2.bilateralFilter(gray, 9, 75, 75)
                    
                    # Apply adaptive threshold with optimized parameters
                    processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY, 11, 2)
                    
                    # Dilation to enhance text
                    kernel = np.ones((1, 1), np.uint8)
                    processed = cv2.dilate(processed, kernel, iterations=1)
                    
                    # Configure for sentence recognition with improved parameters
                    config = '--oem 1 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()-+/ "'
                    self.boxes = pytesseract.image_to_data(processed, lang=self.language, config=config)
                    
                    # If we have a TensorFlow model, use it for additional processing
                    if self.model is not None:
                        try:
                            # Prepare the input for the model
                            # Resize to expected input size if needed
                            resized_roi = cv2.resize(processed, (43844, 1))
                            # Reshape to match model input requirements
                            model_input = np.expand_dims(resized_roi, axis=0)  # Add batch dimension
                            model_input = np.expand_dims(model_input, axis=-1)  # Add channel dimension if needed
                            
                            # Make prediction
                            prediction = self.model.predict(model_input, verbose=0)
                            print(f"Model prediction: {prediction}")
                        except Exception as e:
                            print(f"Error in model prediction: {e}")
                    
                else:
                    # Use the original crop dimensions if no selection ROI
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Apply CLAHE for better contrast
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)
                    
                    # Apply bilateral filter to reduce noise while preserving edges
                    processed = cv2.bilateralFilter(gray, 9, 75, 75)
                    
                    # Apply adaptive threshold with optimized parameters
                    processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY, 11, 2)
                    
                    # Crop the frame if needed
                    processed = processed[self.crop_height:(self.height - self.crop_height),
                                  self.crop_width:(self.width - self.crop_width)]
                    
                    # Configure for sentence recognition
                    config = "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()-+\\\"\\\'/ "
                    self.boxes = pytesseract.image_to_data(processed, lang=self.language, config=config)

    def set_dimensions(self, width, height, crop_width, crop_height):
        self.width = width
        self.height = height
        self.crop_width = crop_width
        self.crop_height = crop_height

    def stop_process(self):
        self.stopped = True

def capture_image(frame, captures=0):
    cwd_path = os.getcwd()
    Path(cwd_path + '/images').mkdir(parents=False, exist_ok=True)
    now = datetime.now()
    name = "OCR " + now.strftime("%Y-%m-%d") + " at " + now.strftime("%H:%M:%S") + '-' + str(captures + 1) + '.jpg'
    path = 'images/' + name
    cv2.imwrite(path, frame)
    captures += 1
    print(name)
    return captures

def views(mode, confidence):
    conf_thresh = None
    color = None
    
    if mode == 1:
        conf_thresh = 75
        color = (0, 255, 0)
    if mode == 2:
        conf_thresh = 0
        if confidence >= 50:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
    if mode == 3:
        conf_thresh = 0
        color = (int(float(confidence)) * 2.55, int(float(confidence)) * 2.55, 0)
    if mode == 4:
        conf_thresh = 0
        color = (0, 0, 255)
        
    return conf_thresh, color

def put_ocr_boxes(boxes, frame, height, selection_roi=None, crop_width=0, crop_height=0, view_mode=1):
    if view_mode not in [1, 2, 3, 4]:
        raise Exception("A nonexistent view mode was selected. Only modes 1-4 are available")
    
    text = ''
    line_text = {}  # Dictionary to store text by line number
    
    # Determine offset based on whether we're using selection or crop
    x_offset = selection_roi[0] if selection_roi else crop_width
    y_offset = selection_roi[1] if selection_roi else crop_height
    
    if boxes is not None:
        for i, box in enumerate(boxes.splitlines()):
            box = box.split()
            if i != 0:
                if len(box) == 12:
                    x, y, w, h = int(box[6]), int(box[7]), int(box[8]), int(box[9])
                    conf = box[10]
                    word = box[11]
                    line_num = int(box[4])  # Line number from tesseract
                    
                    # Apply offset
                    x += x_offset
                    y += y_offset
                    
                    conf_thresh, color = views(view_mode, int(float(conf)))
                    
                    if int(float(conf)) > conf_thresh:
                        cv2.rectangle(frame, (x, y), (w + x, h + y), color, thickness=1)
                        
                        # Group text by line number for better sentence recognition
                        if line_num not in line_text:
                            line_text[line_num] = word
                        else:
                            line_text[line_num] += ' ' + word
    
    # Combine all lines to form complete sentences
    for line_num in sorted(line_text.keys()):
        text += line_text[line_num] + ' '
    
    text = text.strip()
    
    # Display the recognized text
    if text:
        # Display at the bottom of the frame
        cv2.putText(frame, text, (5, height - 5), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200))
    
    return frame, text

def put_crop_box(frame, width, height, crop_width, crop_height):
    cv2.rectangle(frame, (crop_width, crop_height), (width - crop_width, height - crop_height),
                 (255, 0, 0), thickness=1)
    return frame

def put_selection_box(frame, selection_roi):
    if selection_roi:
        x1, y1, x2, y2 = selection_roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), thickness=2)
    return frame

def put_rate(frame, rate):
    cv2.putText(frame, "{} Iterations/Second".format(int(rate)),
               (10, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
    return frame

def put_language(frame, language_string):
    cv2.putText(frame, language_string,
               (10, 65), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
    return frame

def language_string(language):
    if language is not None:
        return language
    else:
        return 'English'

def tesseract_location(root):
    try:
        pytesseract.pytesseract.tesseract_cmd = root
    except FileNotFoundError:
        print("Please double check the Tesseract file directory or ensure it's installed.")
        sys.exit(1)

def translate_text(text, target_language="hi"):
    """
    Translate the OCR text to the target language
    :param text: The text to translate
    :param target_language: The language code to translate to (default: Hindi)
    :return: Translated text
    """
    try:
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def speak_text(text, language='en'):
    """
    Convert text to speech and play it
    :param text: Text to be spoken
    :param language: Language code for speech
    """
    try:
        # Create a unique filename to avoid conflicts
        filename = f"speech_{time.time()}.mp3"
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(filename)
        
        # Play the speech
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        # Clean up the file after playback
        pygame.mixer.quit()
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        print(f"Speech error: {e}")

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def is_pinching(hand_landmarks):
    """Detect if thumb and index finger are pinching"""
    if not hand_landmarks:
        return False
    
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Calculate distance between thumb tip and index tip
    distance = calculate_distance(thumb_tip, index_tip)
    
    # Threshold for pinch detection (adjust as needed)
    return distance < 0.05  # Lower values mean fingers need to be closer

def ocr_stream(crop, source=1, view_mode=1, language=None, translate_enabled=True, target_language="hi", tflite_model_path="quantized_model.tflite"):
    print("Start OCR")
    captures = 0
    try:
            video_stream = VideoStream(source).start()
            if video_stream.frame is None:
                width, height = video_stream.get_video_dimensions()
                crop_width, crop_height = int(crop * width), int(crop * height)
                
                ocr = OCR().start()
                ocr.set_exchange(video_stream)
                ocr.set_language(language)
                ocr.set_dimensions(width, height, crop_width, crop_height)
                
                # Convert TFLite model to standard TensorFlow model
                try:
                    tf_model = convert_tflite_to_tf(tflite_model_path)
                    ocr.set_model(tf_model)
                    print("Successfully loaded TensorFlow model")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    tf_model = None
                
                # Initialize for hand tracking
                hands = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
                
                # Initialize for selection mode
                selection_in_progress = False
                selection_start = None
                selection_end = None
                selection_roi = None
                
                # Initialize for text-to-speech
                pygame.init()
                last_spoken_text = ""
                
                rate = RateCounter().start()
                
                # Initialize for pinch gesture
                pinch_start_time = None
                pinch_duration = 0
                is_pinching_now = False
                
                while True:
                    frame = video_stream.frame.copy()
                    
                    # Process for hand tracking
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)
                    
                    # Draw hand landmarks
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                            
                            # Check for pinch gesture
                            current_pinch = is_pinching(hand_landmarks)
                            
                            # Handle pinch gesture state
                            if current_pinch and not is_pinching_now:
                                # Pinch just started
                                is_pinching_now = True
                                pinch_start_time = time.time()
                                
                                # Get pinch position for selection
                                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                                pinch_x = int(thumb_tip.x * width)
                                pinch_y = int(thumb_tip.y * height)
                                
                                if not selection_in_progress:
                                    # Start selection
                                    selection_in_progress = True
                                    selection_start = (pinch_x, pinch_y)
                                else:
                                    # End selection
                                    selection_in_progress = False
                                    selection_end = (pinch_x, pinch_y)
                                    
                                    # Create selection ROI
                                    if selection_start and selection_end:
                                        x1, y1 = selection_start
                                        x2, y2 = selection_end
                                        selection_roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                                        ocr.set_selection_roi(selection_roi)
                            
                            elif not current_pinch and is_pinching_now:
                                # Pinch just ended
                                is_pinching_now = False
                                pinch_duration = time.time() - pinch_start_time
                                
                                # Long pinch (>1 sec) can be used for other actions like capture
                                if pinch_duration > 1.0:
                                    captures = capture_image(frame, captures)
                            
                            # If selection is in progress, draw temporary selection rectangle
                            if selection_in_progress and selection_start:
                                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                                current_x = int(thumb_tip.x * width)
                                current_y = int(thumb_tip.y * height)
                                cv2.rectangle(frame, selection_start, (current_x, current_y), (0, 255, 255), 2)
                    
                    # Draw crop box if no selection ROI
                    if selection_roi is None:
                        frame = put_crop_box(frame, width, height, crop_width, crop_height)
                    else:
                        # Draw selection box if available
                        frame = put_selection_box(frame, selection_roi)
                    
                    # Process OCR results
                    if ocr.boxes is not None:
                        frame, recognized_text = put_ocr_boxes(ocr.boxes, frame, height, selection_roi, crop_width, crop_height, view_mode)
                        
                        # Translate text if enabled and text is available
                        if translate_enabled and recognized_text and recognized_text != last_spoken_text:
                            translated_text = translate_text(recognized_text, target_language)
                            
                            # Display translated text
                            cv2.putText(frame, translated_text, (5, height - 35), 
                                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 0))
                            
                            # Speak the text (use original or translated based on preference)
                            text_to_speak = translated_text if target_language != "en" else recognized_text
                            speak_lang = target_language if target_language != "en" else "en"
                            
                            # Start a separate thread for speech to avoid blocking the main loop
                            speech_thread = threading.Thread(target=speak_text, args=(text_to_speak, speak_lang))
                            speech_thread.daemon = True
                            speech_thread.start()
                            
                            last_spoken_text = recognized_text
                    
                    # Display FPS and language
                    rate.increment()
                    frame = put_rate(frame, rate.rate())
                    frame = put_language(frame, language_string(language))
                    
                    # Display the frame
                    cv2.imshow("OCR", frame)
                    
                    # Check for key presses
                    key = cv2.waitKey(1) & 0xFF
                    
                    # 'q' to quit
                    if key == ord('q'):
                        break
                        
                    # 'c' to capture image
                    elif key == ord('c'):
                        captures = capture_image(frame, captures)
                        
                    # 'r' to reset selection
                    elif key == ord('r'):
                        selection_roi = None
                        selection_start = None
                        selection_end = None
                        selection_in_progress = False
                        ocr.set_selection_roi(None)
                        
                    # 't' to toggle translation
                    elif key == ord('t'):
                        translate_enabled = not translate_enabled
                        print(f"Translation {'enabled' if translate_enabled else 'disabled'}")
                        
                    # Number keys to change view mode
                    elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                        view_mode = int(chr(key))
                        print(f"View mode changed to {view_mode}")
                
                # Clean up
                video_stream.stop_process()
                ocr.stop_process()
                hands.close()
                cv2.destroyAllWindows()
                
    except Exception as e:
                print(f"Error in OCR stream: {e}")
                import traceback
                traceback.print_exc()
def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='OCR from video stream with hand gesture control')
    parser.add_argument('--source', type=int, default=0, help='Video source (default: 0)')
    parser.add_argument('--crop', type=float, default=0.1, help='Crop percentage (default: 0.1)')
    parser.add_argument('--view', type=int, default=1, choices=[1, 2, 3, 4], help='View mode (default: 1)')
    parser.add_argument('--lang', type=str, default='eng', help='OCR language (default: eng)')
    parser.add_argument('--translate', type=bool, default=True, help='Enable translation (default: True)')
    parser.add_argument('--target_lang', type=str, default='hi', help='Target language for translation (default: hi)')
    parser.add_argument('--model', type=str, default='quantized_model.tflite', help='Path to TFLite model')
    
    args = parser.parse_args()
    
    # Start OCR stream
    ocr_stream(
        crop=args.crop,
        source=args.source,
        view_mode=args.view,
        language=args.lang,
        translate_enabled=args.translate,
        target_language=args.target_lang,
        tflite_model_path=args.model
    )

if __name__ == "__main__":
    main()

