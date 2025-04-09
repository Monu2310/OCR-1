# import os
# import sys
# import time
# import threading
# from threading import Thread
# import cv2
# import numpy as np
# import pytesseract
# import mediapipe as mp
# from datetime import datetime
# from pathlib import Path
# from gtts import gTTS
# import pygame
# from googletrans import Translator

# # Set Tesseract path
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
# os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata/'

# # Initialize MediaPipe Hand solution
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# # Initialize translator
# translator = Translator()

# class RateCounter:
#     def __init__(self):
#         self.start_time = None
#         self.iterations = 0

#     def start(self):
#         self.start_time = time.perf_counter()
#         return self

#     def increment(self):
#         self.iterations += 1

#     def rate(self):
#         elapsed_time = (time.perf_counter() - self.start_time)
#         return self.iterations / elapsed_time

# class VideoStream:
#     def __init__(self, src=0):
#         self.stream = cv2.VideoCapture(src)
#         (self.grabbed, self.frame) = self.stream.read()
#         self.stopped = False

#     def start(self):
#         Thread(target=self.get, args=()).start()
#         return self

#     def get(self):
#         while not self.stopped:
#             (self.grabbed, self.frame) = self.stream.read()

#     def get_video_dimensions(self):
#         width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
#         height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
#         return int(width), int(height)

#     def stop_process(self):
#         self.stopped = True

# class OCR:
#     def __init__(self):
#         self.boxes = None
#         self.stopped = False
#         self.exchange = None
#         self.language = None
#         self.width = None
#         self.height = None
#         self.crop_width = None
#         self.crop_height = None
#         self.frame_count = 0
#         self.process_every_n_frames = 2  # Process every 2nd frame for better performance

#     def start(self):
#         Thread(target=self.ocr, args=()).start()
#         return self

#     def set_exchange(self, video_stream):
#         self.exchange = video_stream

#     def set_language(self, language):
#         self.language = language

#     def ocr(self):
#         while not self.stopped:
#             if self.exchange is not None:
#                 self.frame_count += 1
#                 if self.frame_count % self.process_every_n_frames != 0:
#                     continue
                    
#                 frame = self.exchange.frame
                
#                 # Enhanced image processing for better OCR
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
#                 # Apply bilateral filter to reduce noise while preserving edges
#                 processed = cv2.bilateralFilter(gray, 9, 75, 75)
                
#                 # Apply adaptive threshold with optimized parameters
#                 processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                               cv2.THRESH_BINARY, 11, 2)
                
#                 # Crop the frame if needed
#                 processed = processed[self.crop_height:(self.height - self.crop_height),
#                               self.crop_width:(self.width - self.crop_width)]
                
#                 # Configure for sentence recognition
#                 config = "--oem 1 --psm 6"  # PSM 6 assumes a single uniform block of text
#                 self.boxes = pytesseract.image_to_data(processed, lang=self.language, config=config)

#     def set_dimensions(self, width, height, crop_width, crop_height):
#         self.width = width
#         self.height = height
#         self.crop_width = crop_width
#         self.crop_height = crop_height

#     def stop_process(self):
#         self.stopped = True

# def capture_image(frame, captures=0):
#     cwd_path = os.getcwd()
#     Path(cwd_path + '/images').mkdir(parents=False, exist_ok=True)
#     now = datetime.now()
#     name = "OCR " + now.strftime("%Y-%m-%d") + " at " + now.strftime("%H:%M:%S") + '-' + str(captures + 1) + '.jpg'
#     path = 'images/' + name
#     cv2.imwrite(path, frame)
#     captures += 1
#     print(name)
#     return captures

# def views(mode, confidence):
#     conf_thresh = None
#     color = None
    
#     if mode == 1:
#         conf_thresh = 75
#         color = (0, 255, 0)
#     if mode == 2:
#         conf_thresh = 0
#         if confidence >= 50:
#             color = (0, 255, 0)
#         else:
#             color = (0, 0, 255)
#     if mode == 3:
#         conf_thresh = 0
#         color = (int(float(confidence)) * 2.55, int(float(confidence)) * 2.55, 0)
#     if mode == 4:
#         conf_thresh = 0
#         color = (0, 0, 255)
        
#     return conf_thresh, color

# def put_ocr_boxes(boxes, frame, height, crop_width=0, crop_height=0, view_mode=1, roi_mask = None, roi_coords=None):
#     if view_mode not in [1, 2, 3, 4]:
#         raise Exception("A nonexistent view mode was selected. Only modes 1-4 are available")
    
#     text = ''
#     line_text = {}  # Dictionary to store text by line number
    
#     if boxes is not None:
#         for i, box in enumerate(boxes.splitlines()):
#             box = box.split()
#             if i != 0:
#                 if len(box) == 12:
#                     x, y, w, h = int(box[6]), int(box[7]), int(box[8]), int(box[9])
#                     conf = box[10]
#                     word = box[11]
#                     line_num = int(box[4])  # Line number from tesseract
                    
#                     x += crop_width
#                     y += crop_height
                    
#                      #Skip if ROI is defined and the box is outside the ROI
#                     if roi_coords is not None:
#                         box_center_x = x + w//2
#                         box_center_y = y + h//2
#                         if not (roi_coords[0] <= box_center_x <= roi_coords[2] and 
#                                 roi_coords[1] <= box_center_y <= roi_coords[3]):
#                             continue
#                     conf_thresh, color = views(view_mode, int(float(conf)))
                    
#                     if int(float(conf)) > conf_thresh:
#                         cv2.rectangle(frame, (x, y), (w + x, h + y), color, thickness=1)
                        
#                         # Group text by line number for better sentence recognition
#                         if line_num not in line_text:
#                             line_text[line_num] = word
#                         else:
#                             line_text[line_num] += ' ' + word
    
#     # Combine all lines to form complete sentences
#     for line_num in sorted(line_text.keys()):
#         text += line_text[line_num] + ' '
    
#     text = text.strip()
    
#     if text and text.isascii():
#         cv2.putText(frame, text, (5, height - 5), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200))
    
#     return frame, text

# def put_crop_box(frame, width, height, crop_width, crop_height):
#     cv2.rectangle(frame, (crop_width, crop_height), (width - crop_width, height - crop_height),
#                  (255, 0, 0), thickness=1)
#     return frame

# def put_rate(frame, rate):
#     cv2.putText(frame, "{} Iterations/Second".format(int(rate)),
#                (10, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
#     return frame

# def put_language(frame, language_string):
#     cv2.putText(frame, language_string,
#                (10, 65), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
#     return frame

# def language_string(language):
#     if language is not None:
#         return language
#     else:
#         return 'English'

# def tesseract_location(root):
#     try:
#         pytesseract.pytesseract.tesseract_cmd = root
#     except FileNotFoundError:
#         print("Please double check the Tesseract file directory or ensure it's installed.")
#         sys.exit(1)

# def translate_text(text, target_language="hi"):
#     """
#     Translate the OCR text to the target language
#     :param text: The text to translate
#     :param target_language: The language code to translate to (default: Hindi)
#     :return: Translated text
#     """
#     try:
#         translated = translator.translate(text, dest=target_language)
#         return translated.text
#     except Exception as e:
#         print(f"Translation error: {e}")
#         return text

# def speak_text(text, language='en'):
#     """
#     Convert text to speech and play it
#     :param text: Text to be spoken
#     :param language: Language code for speech
#     """
#     try:
#         # Create a unique filename to avoid conflicts
#         filename = f"speech_{time.time()}.mp3"
#         tts = gTTS(text=text, lang=language, slow=False)
#         tts.save(filename)
        
#         # Play the speech
#         pygame.mixer.init()
#         pygame.mixer.music.load(filename)
#         pygame.mixer.music.play()
#         while pygame.mixer.music.get_busy():
#             pygame.time.Clock().tick(10)
            
#         # Clean up the file after playback
#         pygame.mixer.quit()
#         if os.path.exists(filename):
#             os.remove(filename)
#     except Exception as e:
#         print(f"Speech error: {e}")

# def ocr_stream(crop, source=0, view_mode=1, language=None, translate_enabled=True, target_language="hi"):
#     try:
#         captures = 0
#         video_stream = VideoStream(source).start()
#         img_wi, img_hi = video_stream.get_video_dimensions()
        
#         if crop is None:
#             cropx, cropy = (200, 200)
#         else:
#             cropx, cropy = crop[0], crop[1]
#             if cropx > img_wi or cropy > img_hi or cropx < 0 or cropy < 0:
#                 cropx, cropy = 0, 0
#                 print("Impossible crop dimensions supplied. Dimensions reverted to 0 0")
#         # Add variables for rectangle drawing
#         start_point = None
#         end_point = None
#         rectangle_active = False
#         drawing_mode = False
#         roi_defined = False
#         roi_coords = None
        
#         # Function to detect pinch gesture
#         def detect_pinch(hand_landmarks):
#             thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#             index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
#             # Calculate distance between thumb and index finger tips
#             distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            
#             # Threshold for pinch detection
#             return distance < 0.05
        
#         # Main display loop
#         print("\nPUSH c TO CAPTURE AN IMAGE. PUSH q TO EXIT.")
#         print("PUSH t TO TOGGLE TRANSLATION. PUSH l TO SWITCH LANGUAGE (HINDI/PUNJABI).")
#         print("PUSH s TO SPEAK THE TEXT.")
#         print("PUSH d TO TOGGLE DRAWING MODE.")
#         print("USE THUMBS UP GESTURE TO START/STOP OCR RECOGNITION.")
#         print("USE PINCH GESTURE TO DRAW A RECTANGLE FOR OCR AREA.\n")
        
#         while True:
#             # Existing code for key handling...
#             pressed_key = cv2.waitKey(1) & 0xFF
#             # Add drawing mode toggle with 'd' key
#             if pressed_key == ord('d'):
#                 drawing_mode = not drawing_mode
#                 if drawing_mode:
#                     print("Drawing mode activated. Use pinch gesture to draw rectangle.")
#                 else:
#                     print("Drawing mode deactivated.")
            
#             frame = video_stream.frame
#             hands = mp_hands.Hands(
#                 model_complexity=0,
#                 min_detection_confidence=0.5,
#                 min_tracking_confidence=0.5,
#                 max_num_hands=2)
#             # Convert the BGR image to RGB for MediaPipe
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(rgb_frame)
            
#             # Check for hand gestures
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     # Draw hand landmarks
#                     mp_drawing.draw_landmarks(
#                         frame,
#                         hand_landmarks,
#                         mp_hands.HAND_CONNECTIONS,
#                         mp_drawing_styles.get_default_hand_landmarks_style(),
#                         mp_drawing_styles.get_default_hand_connections_style())

#                         # Check for pinch gesture if in drawing mode
#                     if drawing_mode:
#                         if detect_pinch(hand_landmarks):
#                             # Get coordinates for the pinch point (between thumb and index finger)
#                             h, w, _ = frame.shape
#                             thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#                             index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

#                                 # Calculate midpoint between thumb and index finger
#                             pinch_x = int((thumb_tip.x + index_tip.x) / 2 * w)
#                             pinch_y = int((thumb_tip.y + index_tip.y) / 2 * h)

#                             if not rectangle_active:
#                                 # Start drawing rectangle
#                                 start_point = (pinch_x, pinch_y)
#                                 rectangle_active = True
#                                 print("Started drawing rectangle")
#                             else:
#                                  # Update rectangle endpoint
#                                 end_point = (pinch_x, pinch_y)
#                         elif rectangle_active:
#                                 # Finish drawing rectangle when pinch is released
#                             if start_point and end_point:
#                                     # Ensure correct coordinate order (top-left, bottom-right)
#                                 x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
#                                 x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
#                                 roi_coords = (x1, y1, x2, y2)
#                                 roi_defined = True
#                                 rectangle_active = False
#                                 drawing_mode = False
#                                 print(f"Rectangle defined at coordinates: {roi_coords}")

#                         # Check for thumbs up gesture (existing code)
#                     thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#                     thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

#                         # Thumbs up detection (existing code)
#                     if (thumb_tip.y < thumb_ip.y and 
#                         all(hand_landmarks.landmark[i].y > hand_landmarks.landmark[i-2].y 
#                             for i in [8, 12, 16, 20])):
                            
#                         current_time = time.time()
#                         if current_time - last_gesture_time > gesture_cooldown:
#                             ocr_active = not ocr_active
#                             last_gesture_time = current_time
#                             print(f"OCR {'activated' if ocr_active else 'deactivated'}")
            
#             # Draw rectangle if active or defined
#             if rectangle_active and start_point and end_point:
#                 cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
#             elif roi_defined and roi_coords:
#                 cv2.rectangle(frame, (roi_coords[0], roi_coords[1]), (roi_coords[2], roi_coords[3]), (0, 255, 0), 2)
            
#             ocr_active = False  # OCR starts inactive
#             last_gesture_time = time.time()
#             gesture_cooldown = 1.0 

#             # Display status
#             status_text = "OCR: ACTIVE" if ocr_active else "OCR: INACTIVE"
#             cv2.putText(frame, status_text, (10, 95), cv2.FONT_HERSHEY_DUPLEX, 1.0, 
#                        (0, 255, 0) if ocr_active else (0, 0, 255), 2)
            
#             # Display drawing mode status
#             draw_status = "DRAWING MODE: ON" if drawing_mode else "DRAWING MODE: OFF"
#             cv2.putText(frame, draw_status, (10, 155), cv2.FONT_HERSHEY_DUPLEX, 1.0,
#                         (0, 255, 0) if drawing_mode else (0, 0, 255), 2)
            
#              # Display translation status and language (existing code)
#             if translate_enabled:
#                 if target_language == "hi":
#                     trans_status = "HINDI TRANSLATION: ON"
#                 elif target_language == "pa":
#                     trans_status = "PUNJABI TRANSLATION: ON"
#                 else:
#                     trans_status = f"TRANSLATION ({target_language}): ON"
#             else:
#                 trans_status = "TRANSLATION: OFF"
                
#             cv2.putText(frame, trans_status, (10, 125), cv2.FONT_HERSHEY_DUPLEX, 1.0,
#                         (0, 255, 0) if translate_enabled else (0, 0, 255), 2)
#             cps1 = RateCounter().start()
#             lang_name = language_string(language)
#             # Add other display elements (existing code)
#             frame = put_rate(frame, cps1.rate())
#             frame = put_language(frame, lang_name)
            
#             # Only process OCR if active and within ROI if defined
#             if ocr_active and ocr.boxes is not None:
#                 if roi_defined:
#                     # Create a mask for the ROI
#                     mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#                     cv2.rectangle(mask, (roi_coords[0], roi_coords[1]), (roi_coords[2], roi_coords[3]), 255, -1)
                    
#                     # Apply mask to get only text within ROI
#                     roi_frame, text = put_ocr_boxes(ocr.boxes, frame, img_hi,
#                                                crop_width=cropx, crop_height=cropy, view_mode=view_mode, 
#                                                roi_mask=mask, roi_coords=roi_coords)
#                     frame = roi_frame
#                 else:
#                     frame, text = put_ocr_boxes(ocr.boxes, frame, img_hi,
#                                            crop_width=cropx, crop_height=cropy, view_mode=view_mode)
                
#                 # Print recognized text to terminal if it's different from the last one
#                 if text and text != last_recognized_text:
#                     print("\nRecognized Text:")
#                     print(text)
#                     last_recognized_text = text
                    
#                 # Add translation if enabled (existing code)
#                     if translate_enabled and text:
#                         translated_text = translate_text(text, target_language)
#                     # Display translated text
#                         cv2.putText(frame, translated_text, (5, img_hi - 35), 
#                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 200))
#                         # Print translation to terminal
#                         print(f"Translation ({target_language}):")
#                         print(translated_text)


        
#         # ocr = OCR().start()
#         # print("OCR stream started")
#         # print("Active threads: {}".format(threading.activeCount()))
#         # ocr.set_exchange(video_stream)
#         # ocr.set_language(language)
#         # ocr.set_dimensions(img_wi, img_hi, cropx, cropy)
        
#         # cps1 = RateCounter().start()
#         # lang_name = language_string(language)
        
#         # # Initialize MediaPipe hand tracking
#         # hands = mp_hands.Hands(
#         #     model_complexity=0,
#         #     min_detection_confidence=0.5,
#         #     min_tracking_confidence=0.5,
#         #     max_num_hands=2)
        
#         # # Gesture control variables
#         # ocr_active = False  # OCR starts inactive
#         # last_gesture_time = time.time()
#         # gesture_cooldown = 1.0  # Cooldown in seconds to prevent accidental triggers
        
#         # # Translation and speech variables
#         # translated_text = ""
#         # last_recognized_text = ""
        
#         # # Main display loop
#         # print("\nPUSH c TO CAPTURE AN IMAGE. PUSH q TO EXIT.")
#         # print("PUSH t TO TOGGLE TRANSLATION. PUSH l TO SWITCH LANGUAGE (HINDI/PUNJABI).")
#         # print("PUSH s TO SPEAK THE TEXT.")
#         # print("USE THUMBS UP GESTURE TO START/STOP OCR RECOGNITION.\n")
        
#         # while True:
#         #     # Quit condition:
#         #     pressed_key = cv2.waitKey(1) & 0xFF
#         #     if pressed_key == ord('q'):
#         #         video_stream.stop_process()
#         #         ocr.stop_process()
#         #         print("OCR stream stopped\n")
#         #         print("{} image(s) captured and saved to current directory".format(captures))
#         #         break
            
#         #     frame = video_stream.frame
            
#         #     # Convert the BGR image to RGB for MediaPipe
#         #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         #     results = hands.process(rgb_frame)
            
#         #     # Check for hand gestures
#         #     if results.multi_hand_landmarks:
#         #         for hand_landmarks in results.multi_hand_landmarks:
#         #             # Draw hand landmarks
#         #             mp_drawing.draw_landmarks(
#         #                 frame,
#         #                 hand_landmarks,
#         #                 mp_hands.HAND_CONNECTIONS,
#         #                 mp_drawing_styles.get_default_hand_landmarks_style(),
#         #                 mp_drawing_styles.get_default_hand_connections_style())
                    
#         #             # Check for thumbs up gesture
#         #             thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#         #             thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                    
#         #             # Thumbs up detection: thumb is pointing up and other fingers are down
#         #             if (thumb_tip.y < thumb_ip.y and 
#         #                 all(hand_landmarks.landmark[i].y > hand_landmarks.landmark[i-2].y 
#         #                     for i in [8, 12, 16, 20])):  # Check if other fingers are curled
                        
#         #                 current_time = time.time()
#         #                 if current_time - last_gesture_time > gesture_cooldown:
#         #                     ocr_active = not ocr_active
#         #                     last_gesture_time = current_time
#         #                     print(f"OCR {'activated' if ocr_active else 'deactivated'}")
            
#         #     # Display status
#         #     status_text = "OCR: ACTIVE" if ocr_active else "OCR: INACTIVE"
#         #     cv2.putText(frame, status_text, (10, 95), cv2.FONT_HERSHEY_DUPLEX, 1.0, 
#         #                (0, 255, 0) if ocr_active else (0, 0, 255), 2)
            
#         #     # Display translation status and language
#         #     if translate_enabled:
#         #         if target_language == "hi":
#         #             trans_status = "HINDI TRANSLATION: ON"
#         #         elif target_language == "pa":
#         #             trans_status = "PUNJABI TRANSLATION: ON"
#         #         else:
#         #             trans_status = f"TRANSLATION ({target_language}): ON"
#         #     else:
#         #         trans_status = "TRANSLATION: OFF"
                
#         #     cv2.putText(frame, trans_status, (10, 125), cv2.FONT_HERSHEY_DUPLEX, 1.0,
#         #                (0, 255, 0) if translate_enabled else (0, 0, 255), 2)
            
#         #     # Add other display elements
#         #     frame = put_rate(frame, cps1.rate())
#         #     frame = put_language(frame, lang_name)
#         #     frame = put_crop_box(frame, img_wi, img_hi, cropx, cropy)
            
#         #     # Only process OCR if active
#         #     if ocr_active and ocr.boxes is not None:
#         #         frame, text = put_ocr_boxes(ocr.boxes, frame, img_hi,
#         #                                    crop_width=cropx, crop_height=cropy, view_mode=view_mode)
                
#         #         # Print recognized text to terminal if it's different from the last one
#         #         if text and text != last_recognized_text:
#         #             print("\nRecognized Text:")
#         #             print(text)
#         #             last_recognized_text = text
                    
#         #             # Add translation if enabled
#         #             if translate_enabled and text:
#         #                 translated_text = translate_text(text, target_language)
#         #                 # Display translated text
#         #                 cv2.putText(frame, translated_text, (5, img_hi - 35), 
#         #                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 200))
#         #                 # Print translation to terminal
#         #                 print(f"Translation ({target_language}):")
#         #                 print(translated_text)
            
#         #     # Toggle translation with 't' key
#         #     if pressed_key == ord('t'):
#         #         translate_enabled = not translate_enabled
#         #         print(f"Translation {'enabled' if translate_enabled else 'disabled'}")
            
#         #     # Switch between Hindi and Punjabi with 'l' key
#         #     if pressed_key == ord('l'):
#         #         if target_language == "hi":
#         #             target_language = "pa"
#         #             print("Switched to Punjabi translation")
#         #         else:
#         #             target_language = "hi"
#         #             print("Switched to Hindi translation")
                
#         #         # Update translation if text exists
#         #         if ocr_active and 'text' in locals() and text and translate_enabled:
#         #             translated_text = translate_text(text, target_language)
            
#         #     # Speak text with 's' key
#         #     if pressed_key == ord('s'):
#         #         if ocr_active and 'text' in locals() and text:
#         #             print("Attempting to speak text...")
#         #             if translate_enabled and translated_text:
#         #                 Thread(target=speak_text, args=(translated_text, target_language)).start()
#         #                 print(f"Speaking translated text: {translated_text}")
#         #             else:
#         #                 Thread(target=speak_text, args=(text, 'en')).start()
#         #                 print(f"Speaking text: {text}")
            
#         #     # Photo capture:
#         #     if pressed_key == ord('c'):
#         #         if ocr_active and 'text' in locals():
#         #             print('\n' + text)
#         #             if translate_enabled and translated_text:
#         #                 print(f"Translation: {translated_text}")
#         #         captures = capture_image(frame, captures)
            
#         #     cv2.imshow("Real-time OCR", frame)
#         #     cps1.increment()
        
#         # Clean up
#         hands.close()
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"Error in OCR stream: {e}")
#         import traceback
#         traceback.print_exc()
# # # Add variables for rectangle drawing
# #     start_point = None
# #     end_point = None
# #     rectangle_active = False
# #     drawing_mode = False
# #     roi_defined = False
# #     roi_coords = None
        
# #         # Function to detect pinch gesture
# #     def detect_pinch(hand_landmarks):
# #         thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
# #         index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
# #             # Calculate distance between thumb and index finger tips
# #         distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            
# #             # Threshold for pinch detection
# #         return distance < 0.05
        
# #         # Main display loop
# #     print("\nPUSH c TO CAPTURE AN IMAGE. PUSH q TO EXIT.")
# #     print("PUSH t TO TOGGLE TRANSLATION. PUSH l TO SWITCH LANGUAGE (HINDI/PUNJABI).")
# #     print("PUSH s TO SPEAK THE TEXT.")
# #     print("PUSH d TO TOGGLE DRAWING MODE.")
# #     print("USE THUMBS UP GESTURE TO START/STOP OCR RECOGNITION.")
# #     print("USE PINCH GESTURE TO DRAW A RECTANGLE FOR OCR AREA.\n")
        
# #     while True:
# #             # Existing code for key handling...
            
# #             # Add drawing mode toggle with 'd' key
# #         if pressed_key == ord('d'):
# #             drawing_mode = not drawing_mode
# #             if drawing_mode:
# #                 print("Drawing mode activated. Use pinch gesture to draw rectangle.")
# #             else:
# #                 print("Drawing mode deactivated.")
            
# #         frame = video_stream.frame
            
# #             # Convert the BGR image to RGB for MediaPipe
# #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         results = hands.process(rgb_frame)
            
# #             # Check for hand gestures
# #         if results.multi_hand_landmarks:
# #             for hand_landmarks in results.multi_hand_landmarks:
# #                 # Draw hand landmarks
# #                 mp_drawing.draw_landmarks(
# #                     frame,
# #                     hand_landmarks,
# #                     mp_hands.HAND_CONNECTIONS,
# #                     mp_drawing_styles.get_default_hand_landmarks_style(),
# #                     mp_drawing_styles.get_default_hand_connections_style())
                    
# #                     # Check for pinch gesture if in drawing mode
# #                 if drawing_mode:
# #                     if detect_pinch(hand_landmarks):
# #                         # Get coordinates for the pinch point (between thumb and index finger)
# #                         h, w, _ = frame.shape
# #                         thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
# #                         index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            
# #                             # Calculate midpoint between thumb and index finger
# #                         pinch_x = int((thumb_tip.x + index_tip.x) / 2 * w)
# #                         pinch_y = int((thumb_tip.y + index_tip.y) / 2 * h)
                            
# #                         if not rectangle_active:
# #                             # Start drawing rectangle
# #                             start_point = (pinch_x, pinch_y)
# #                             rectangle_active = True
# #                             print("Started drawing rectangle")
# #                         else:
# #                              # Update rectangle endpoint
# #                             end_point = (pinch_x, pinch_y)
# #                     elif rectangle_active:
# #                             # Finish drawing rectangle when pinch is released
# #                         if start_point and end_point:
# #                                 # Ensure correct coordinate order (top-left, bottom-right)
# #                             x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
# #                             x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
# #                             roi_coords = (x1, y1, x2, y2)
# #                             roi_defined = True
# #                             rectangle_active = False
# #                             drawing_mode = False
# #                             print(f"Rectangle defined at coordinates: {roi_coords}")
                    
# #                     # Check for thumbs up gesture (existing code)
# #                 thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
# #                 thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                    
# #                     # Thumbs up detection (existing code)
# #                 if (thumb_tip.y < thumb_ip.y and 
# #                     all(hand_landmarks.landmark[i].y > hand_landmarks.landmark[i-2].y 
# #                         for i in [8, 12, 16, 20])):
                        
# #                     current_time = time.time()
# #                     if current_time - last_gesture_time > gesture_cooldown:
# #                         ocr_active = not ocr_active
# #                         last_gesture_time = current_time
# #                         print(f"OCR {'activated' if ocr_active else 'deactivated'}")
            
# #             # Draw rectangle if active or defined
# #         if rectangle_active and start_point and end_point:
# #             cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
# #         elif roi_defined and roi_coords:
# #             cv2.rectangle(frame, (roi_coords[0], roi_coords[1]), (roi_coords[2], roi_coords[3]), (0, 255, 0), 2)
            
# #             # Display status
# #         status_text = "OCR: ACTIVE" if ocr_active else "OCR: INACTIVE"
# #         cv2.putText(frame, status_text, (10, 95), cv2.FONT_HERSHEY_DUPLEX, 1.0, 
# #                     (0, 255, 0) if ocr_active else (0, 0, 255), 2)
            
# #             # Display drawing mode status
# #         draw_status = "DRAWING MODE: ON" if drawing_mode else "DRAWING MODE: OFF"
# #         cv2.putText(frame, draw_status, (10, 155), cv2.FONT_HERSHEY_DUPLEX, 1.0,
# #                     (0, 255, 0) if drawing_mode else (0, 0, 255), 2)
            
# #             # Display translation status and language (existing code)
# #         if translate_enabled:
# #             if target_language == "hi":
# #                 trans_status = "HINDI TRANSLATION: ON"
# #             elif target_language == "pa":
# #                 trans_status = "PUNJABI TRANSLATION: ON"
# #             else:
# #                 trans_status = f"TRANSLATION ({target_language}): ON"
# #         else:
# #             trans_status = "TRANSLATION: OFF"
                
# #         cv2.putText(frame, trans_status, (10, 125), cv2.FONT_HERSHEY_DUPLEX, 1.0,
# #                     (0, 255, 0) if translate_enabled else (0, 0, 255), 2)
            
# #             # Add other display elements (existing code)
# #         frame = put_rate(frame, cps1.rate())
# #         frame = put_language(frame, lang_name)
            
# #             # Only process OCR if active and within ROI if defined
# #         if ocr_active and ocr.boxes is not None:
# #             if roi_defined:
# #                     # Create a mask for the ROI
# #                 mask = np.zeros(frame.shape[:2], dtype=np.uint8)
# #                 cv2.rectangle(mask, (roi_coords[0], roi_coords[1]), (roi_coords[2], roi_coords[3]), 255, -1)
                    
# #                     # Apply mask to get only text within ROI
# #                 roi_frame, text = put_ocr_boxes(ocr.boxes, frame, img_hi,
# #                                            crop_width=cropx, crop_height=cropy, view_mode=view_mode, 
# #                                            roi_mask=mask, roi_coords=roi_coords)
# #                 frame = roi_frame
# #             else:
# #                 frame, text = put_ocr_boxes(ocr.boxes, frame, img_hi,
# #                                            crop_width=cropx, crop_height=cropy, view_mode=view_mode)
                
# #                 # Print recognized text to terminal if it's different from the last one
# #             if text and text != last_recognized_text:
# #                 print("\nRecognized Text:")
# #                 print(text)
# #                 last_recognized_text = text
                    
# #                 # Add translation if enabled (existing code)
# #                 if translate_enabled and text:
# #                     translated_text = translate_text(text, target_language)
# #                     # Display translated text
# #                     cv2.putText(frame, translated_text, (5, img_hi - 35), 
# #                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 200))
# #                         # Print translation to terminal
# #                     print(f"Translation ({target_language}):")
# #                     print(translated_text)
            
#             # Rest of the existing code...
            
#         cv2.imshow("Real-time OCR", frame)
#         cps1.increment()

# if __name__ == '__main__':
#     try:
#         tess_path = '/opt/homebrew/bin/tesseract'  
#         view_mode = 1
#         source = 0
#         crop = [100, 100]
#         language = "eng"
#         translate_enabled = True
#         target_language = "hi"  # Default to Hindi
        
#         tesseract_location(tess_path)
#         ocr_stream(view_mode=view_mode, source=source, crop=crop, language=language,
#                   translate_enabled=translate_enabled, target_language=target_language)
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback

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

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata/'

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize translator
translator = Translator()

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

    def start(self):
        Thread(target=self.ocr, args=()).start()
        return self

    def set_exchange(self, video_stream):
        self.exchange = video_stream

    def set_language(self, language):
        self.language = language

    def ocr(self):
        while not self.stopped:
            if self.exchange is not None:
                self.frame_count += 1
                if self.frame_count % self.process_every_n_frames != 0:
                    continue
                    
                frame = self.exchange.frame
                
                # Enhanced image processing for better OCR
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply bilateral filter to reduce noise while preserving edges
                processed = cv2.bilateralFilter(gray, 9, 75, 75)
                
                # Apply adaptive threshold with optimized parameters
                processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
                
                # Crop the frame if needed
                processed = processed[self.crop_height:(self.height - self.crop_height),
                              self.crop_width:(self.width - self.crop_width)]
                
                # Configure for sentence recognition
                config = "--oem 1 --psm 6"  # PSM 6 assumes a single uniform block of text
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

def put_ocr_boxes(boxes, frame, height, crop_width=0, crop_height=0, view_mode=1):
    if view_mode not in [1, 2, 3, 4]:
        raise Exception("A nonexistent view mode was selected. Only modes 1-4 are available")
    
    text = ''
    line_text = {}  # Dictionary to store text by line number
    
    if boxes is not None:
        for i, box in enumerate(boxes.splitlines()):
            box = box.split()
            if i != 0:
                if len(box) == 12:
                    x, y, w, h = int(box[6]), int(box[7]), int(box[8]), int(box[9])
                    conf = box[10]
                    word = box[11]
                    line_num = int(box[4])  # Line number from tesseract
                    
                    x += crop_width
                    y += crop_height
                    
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
    
    if text and text.isascii():
        cv2.putText(frame, text, (5, height - 5), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200))
    
    return frame, text

def put_crop_box(frame, width, height, crop_width, crop_height):
    cv2.rectangle(frame, (crop_width, crop_height), (width - crop_width, height - crop_height),
                 (255, 0, 0), thickness=1)
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

def ocr_stream(crop, source=0, view_mode=1, language=None, translate_enabled=True, target_language="hi"):
    try:
        captures = 0
        video_stream = VideoStream(source).start()
        img_wi, img_hi = video_stream.get_video_dimensions()
        
        if crop is None:
            cropx, cropy = (200, 200)
        else:
            cropx, cropy = crop[0], crop[1]
            if cropx > img_wi or cropy > img_hi or cropx < 0 or cropy < 0:
                cropx, cropy = 0, 0
                print("Impossible crop dimensions supplied. Dimensions reverted to 0 0")
        
        ocr = OCR().start()
        print("OCR stream started")
        print("Active threads: {}".format(threading.activeCount()))
        ocr.set_exchange(video_stream)
        ocr.set_language(language)
        ocr.set_dimensions(img_wi, img_hi, cropx, cropy)
        
        cps1 = RateCounter().start()
        lang_name = language_string(language)
        
        # Initialize MediaPipe hand tracking
        hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2)
        
        # Gesture control variables
        ocr_active = False  # OCR starts inactive
        last_gesture_time = time.time()
        gesture_cooldown = 1.0  # Cooldown in seconds to prevent accidental triggers
        
        # Translation and speech variables
        translated_text = ""
        last_recognized_text = ""
        
        # Main display loop
        print("\nPUSH c TO CAPTURE AN IMAGE. PUSH q TO EXIT.")
        print("PUSH t TO TOGGLE TRANSLATION. PUSH l TO SWITCH LANGUAGE (HINDI/PUNJABI).")
        print("PUSH s TO SPEAK THE TEXT.")
        print("USE THUMBS UP GESTURE TO START/STOP OCR RECOGNITION.\n")
        
        while True:
            # Quit condition:
            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key == ord('q'):
                video_stream.stop_process()
                ocr.stop_process()
                print("OCR stream stopped\n")
                print("{} image(s) captured and saved to current directory".format(captures))
                break
            
            frame = video_stream.frame
            
            # Convert the BGR image to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Check for hand gestures
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Check for thumbs up gesture
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                    
                    # Thumbs up detection: thumb is pointing up and other fingers are down
                    if (thumb_tip.y < thumb_ip.y and 
                        all(hand_landmarks.landmark[i].y > hand_landmarks.landmark[i-2].y 
                            for i in [8, 12, 16, 20])):  # Check if other fingers are curled
                        
                        current_time = time.time()
                        if current_time - last_gesture_time > gesture_cooldown:
                            ocr_active = not ocr_active
                            last_gesture_time = current_time
                            print(f"OCR {'activated' if ocr_active else 'deactivated'}")
            
            # Display status
            status_text = "OCR: ACTIVE" if ocr_active else "OCR: INACTIVE"
            cv2.putText(frame, status_text, (10, 95), cv2.FONT_HERSHEY_DUPLEX, 1.0, 
                       (0, 255, 0) if ocr_active else (0, 0, 255), 2)
            
            # Display translation status and language
            if translate_enabled:
                if target_language == "hi":
                    trans_status = "HINDI TRANSLATION: ON"
                elif target_language == "pa":
                    trans_status = "PUNJABI TRANSLATION: ON"
                else:
                    trans_status = f"TRANSLATION ({target_language}): ON"
            else:
                trans_status = "TRANSLATION: OFF"
                
            cv2.putText(frame, trans_status, (10, 125), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                       (0, 255, 0) if translate_enabled else (0, 0, 255), 2)
            
            # Add other display elements
            frame = put_rate(frame, cps1.rate())
            frame = put_language(frame, lang_name)
            frame = put_crop_box(frame, img_wi, img_hi, cropx, cropy)
            
            # Only process OCR if active
            if ocr_active and ocr.boxes is not None:
                frame, text = put_ocr_boxes(ocr.boxes, frame, img_hi,
                                           crop_width=cropx, crop_height=cropy, view_mode=view_mode)
                
                # Print recognized text to terminal if it's different from the last one
                if text and text != last_recognized_text:
                    print("\nRecognized Text:")
                    print(text)
                    last_recognized_text = text
                    
                    # Add translation if enabled
                    if translate_enabled and text:
                        translated_text = translate_text(text, target_language)
                        # Display translated text
                        cv2.putText(frame, translated_text, (5, img_hi - 35), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 200))
                        # Print translation to terminal
                        print(f"Translation ({target_language}):")
                        print(translated_text)
            
            # Toggle translation with 't' key
            if pressed_key == ord('t'):
                translate_enabled = not translate_enabled
                print(f"Translation {'enabled' if translate_enabled else 'disabled'}")
            
            # Switch between Hindi and Punjabi with 'l' key
            if pressed_key == ord('l'):
                if target_language == "hi":
                    target_language = "pa"
                    print("Switched to Punjabi translation")
                else:
                    target_language = "hi"
                    print("Switched to Hindi translation")
                
                # Update translation if text exists
                if ocr_active and 'text' in locals() and text and translate_enabled:
                    translated_text = translate_text(text, target_language)
            
            # Speak text with 's' key
            if pressed_key == ord('s'):
                if ocr_active and 'text' in locals() and text:
                    print("Attempting to speak text...")
                    if translate_enabled and translated_text:
                        Thread(target=speak_text, args=(translated_text, target_language)).start()
                        print(f"Speaking translated text: {translated_text}")
                    else:
                        Thread(target=speak_text, args=(text, 'en')).start()
                        print(f"Speaking text: {text}")
            
            # Photo capture:
            if pressed_key == ord('c'):
                if ocr_active and 'text' in locals():
                    print('\n' + text)
                    if translate_enabled and translated_text:
                        print(f"Translation: {translated_text}")
                captures = capture_image(frame, captures)
            
            cv2.imshow("Real-time OCR", frame)
            cps1.increment()
        
        # Clean up
        hands.close()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error in OCR stream: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    try:
        tess_path = '/opt/homebrew/bin/tesseract'  
        view_mode = 1
        source = 0
        crop = [100, 100]
        language = "eng"
        translate_enabled = True
        target_language = "hi"  # Default to Hindi
        
        tesseract_location(tess_path)
        ocr_stream(view_mode=view_mode, source=source, crop=crop, language=language,
                  translate_enabled=translate_enabled, target_language=target_language)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback