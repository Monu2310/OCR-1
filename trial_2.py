# # # import cv2
# # # import torch
# # # import pytesseract
# # # import supervision as sv
# # # from ultralytics import YOLO

# # # # Optional: Set tesseract path (Windows users only)
# # # # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # # # 1. Load YOLO model
# # # original_load = torch.load
# # # torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
# # # model = YOLO('yolov10s-doclaynet.pt')

# # # # 2. Initialize webcam
# # # cap = cv2.VideoCapture(0)
# # # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# # # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# # # # 3. Supervision annotators
# # # box_annotator = sv.BoxAnnotator()
# # # label_annotator = sv.LabelAnnotator()

# # # while True:
# # #     ret, frame = cap.read()
# # #     if not ret:
# # #         break

# # #     processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# # #     results = model(
# # #         processed_frame,
# # #         imgsz=1024,
# # #         conf=0.2,
# # #         iou=0.8
# # #     )[0]

# # #     detections = sv.Detections.from_ultralytics(results)

# # #     # Filter detections for text classes (you can refine this later)
# # #     text_detections = sv.Detections(
# # #         xyxy=detections.xyxy,
# # #         confidence=detections.confidence,
# # #         class_id=detections.class_id
# # #     )

# # #     annotated_frame = frame.copy()
# # #     labels = []

# # #     for i, (xyxy, conf, cls_id) in enumerate(zip(text_detections.xyxy, text_detections.confidence, text_detections.class_id)):
# # #         x1, y1, x2, y2 = map(int, xyxy)
# # #         roi = frame[y1:y2, x1:x2]

# # #         # Preprocess ROI for better OCR (optional)
# # #         gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# # #         _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # #         # Run OCR on the ROI
# # #         extracted_text = pytesseract.image_to_string(thresh_roi, config='--psm 6')  # psm 6: Assume a block of text

# # #         # Clean up text
# # #         extracted_text = extracted_text.strip().replace("\n", " ")

# # #         # Store label
# # #         label = f"{model.names[cls_id]}: {extracted_text if extracted_text else 'No text'}"
# # #         labels.append(label)

# # #     # Draw boxes
# # #     annotated_frame = box_annotator.annotate(annotated_frame, text_detections)
    
# # #     # Draw labels
# # #     if labels:
# # #         annotated_frame = label_annotator.annotate(annotated_frame, text_detections, labels)

# # #     cv2.imshow("YOLO + OCR", annotated_frame)

# # #     if cv2.waitKey(1) == ord('q'):
# # #         break

# # # cap.release()
# # # cv2.destroyAllWindows()

# # import cv2
# # import torch
# # import pytesseract
# # import supervision as sv
# # from ultralytics import YOLO
# # import pyttsx3
# # from googletrans import Translator

# # # Initialize text-to-speech engine
# # tts_engine = pyttsx3.init()

# # # Initialize translator
# # translator = Translator()

# # # 1. Load YOLO model
# # original_load = torch.load
# # torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
# # model = YOLO('yolov10s-doclaynet.pt')

# # # 2. Initialize webcam
# # cap = cv2.VideoCapture(0)
# # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# # # 3. Supervision annotators
# # box_annotator = sv.BoxAnnotator()
# # label_annotator = sv.LabelAnnotator()

# # # Translation language options
# # languages = {
# #     0: {"code": None, "name": "Original"},
# #     1: {"code": "hi", "name": "Hindi"},
# #     2: {"code": "pa", "name": "Punjabi"}
# # }
# # current_language = 0  # Default to original (no translation)
# # speak_text = False  # Flag to control text-to-speech

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     # Resize frame for faster processing
# #     frame = cv2.resize(frame, (640, 480))
# #     processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# #     results = model(
# #         processed_frame,
# #         imgsz=640,  # Reduced image size for faster processing
# #         conf=0.3,   # Increased confidence threshold
# #         iou=0.5     # Adjusted IOU threshold
# #     )[0]

# #     detections = sv.Detections.from_ultralytics(results)

# #     annotated_frame = frame.copy()
# #     labels = []
# #     extracted_texts = []

# #     for i, (xyxy, conf, cls_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
# #         x1, y1, x2, y2 = map(int, xyxy)
# #         roi = frame[y1:y2, x1:x2]

# #         # Preprocess ROI for better OCR
# #         gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# #         _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# #         # Run OCR on the ROI
# #         extracted_text = pytesseract.image_to_string(thresh_roi, config='--psm 6')
        
# #         # Clean up text
# #         extracted_text = extracted_text.strip().replace("\n", " ")
        
# #         if extracted_text:
# #             extracted_texts.append(extracted_text)
            
# #             # Translate text if a language is selected
# #             if current_language > 0:
# #                 try:
# #                     translated = translator.translate(
# #                         extracted_text, 
# #                         dest=languages[current_language]["code"]
# #                     )
# #                     display_text = translated.text
# #                 except Exception as e:
# #                     display_text = extracted_text + " (Translation failed)"
# #             else:
# #                 display_text = extracted_text
                
# #             label = f"{model.names[cls_id]}: {display_text[:30]}..."  # Truncate long text
# #             labels.append(label)

# #     # Ensure labels match detections
# #     if len(labels) != len(detections):
# #         labels = labels[:len(detections)]  # Truncate labels if necessary

# #     # Draw boxes and labels
# #     annotated_frame = box_annotator.annotate(annotated_frame, detections)
# #     if labels:
# #         annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
    
# #     # Display current language and controls
# #     language_info = f"Language: {languages[current_language]['name']} | TTS: {'ON' if speak_text else 'OFF'}"
# #     controls_info = "Press 'l' to change language, 's' to toggle speech, 'r' to read text, 'q' to quit"
    
# #     cv2.putText(annotated_frame, language_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# #     cv2.putText(annotated_frame, controls_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
# #     cv2.imshow("YOLO + OCR + Translation", annotated_frame)
    
# #     key = cv2.waitKey(1)
# #     if key == ord('q'):
# #         break
# #     elif key == ord('l'):
# #         current_language = (current_language + 1) % len(languages)
# #     elif key == ord('s'):
# #         speak_text = not speak_text
# #     elif key == ord('r') and extracted_texts:
# #         if current_language > 0:
# #             try:
# #                 combined_text = " ".join(extracted_texts)
# #                 translated = translator.translate(
# #                     combined_text, 
# #                     dest=languages[current_language]["code"]
# #                 )
# #                 tts_engine.say(translated.text)
# #                 tts_engine.runAndWait()
# #             except Exception as e:
# #                 print(f"Translation error: {e}")
# #         else:
# #             combined_text = " ".join(extracted_texts)
# #             tts_engine.say(combined_text)
# #             tts_engine.runAndWait()

# # cap.release()
# # cv2.destroyAllWindows()

# import cv2
# import torch
# import pytesseract
# import supervision as sv
# from ultralytics import YOLO
# import pyttsx3
# from deep_translator import GoogleTranslator

# # Initialize text-to-speech engine
# tts_engine = pyttsx3.init()

# # 1. Load YOLO model
# original_load = torch.load
# torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
# model = YOLO('yolov10s-doclaynet.pt')

# # 2. Initialize webcam
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# # 3. Supervision annotators
# box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# # Translation language options
# languages = {
#     0: {"code": "en", "name": "Original"},
#     1: {"code": "hi", "name": "Hindi"},
#     2: {"code": "pa", "name": "Punjabi"}
# }
# current_language = 0  # Default to original (no translation)
# speak_text = False  # Flag to control text-to-speech

# # Initialize translator (will be created when needed)
# translator = None

# # Function to get translation
# def translate_text(text, target_language):
#     global translator
#     if not text:
#         return ""
#     try:
#         # Create translator for the target language
#         translator = GoogleTranslator(source='auto', target=target_language)
#         return translator.translate(text)
#     except Exception as e:
#         print(f"Translation error: {e}")
#         return text + " (Translation failed)"

# # Process every n frames to reduce lag
# process_every_n_frames = 5
# frame_count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     frame_count += 1
    
#     # Only process every nth frame
#     if frame_count % process_every_n_frames != 0:
#         # Still show the frame with previous annotations
#         cv2.imshow("YOLO + OCR + Translation", frame)
#         if cv2.waitKey(1) == ord('q'):
#             break
#         continue
    
#     # Resize frame for faster processing
#     frame = cv2.resize(frame, (640, 480))
#     processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = model(
#         processed_frame,
#         imgsz=640,  # Reduced image size for faster processing
#         conf=0.3,   # Increased confidence threshold
#         iou=0.5     # Adjusted IOU threshold
#     )[0]

#     detections = sv.Detections.from_ultralytics(results)

#     annotated_frame = frame.copy()
#     labels = []
#     extracted_texts = []

#     # Skip processing if no detections to save resources
#     if len(detections) > 0:
#         for i, (xyxy, conf, cls_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
#             x1, y1, x2, y2 = map(int, xyxy)
            
#             # Ensure coordinates are within frame boundaries
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
#             # Skip if region is too small
#             if x2 <= x1 or y2 <= y1 or (x2-x1)*(y2-y1) < 100:
#                 labels.append(f"{model.names[cls_id]}: (region too small)")
#                 continue
                
#             roi = frame[y1:y2, x1:x2]

#             # Preprocess ROI for better OCR
#             try:
#                 gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#                 _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#                 # Run OCR on the ROI
#                 extracted_text = pytesseract.image_to_string(thresh_roi, config='--psm 6')
                
#                 # Clean up text
#                 extracted_text = extracted_text.strip().replace("\n", " ")
                
#                 if extracted_text:
#                     extracted_texts.append(extracted_text)
                    
#                     # Translate text if a language is selected
#                     if current_language > 0:
#                         display_text = translate_text(extracted_text, languages[current_language]["code"])
#                     else:
#                         display_text = extracted_text
                        
#                     # Truncate long text for display
#                     display_text_short = display_text[:30] + "..." if len(display_text) > 30 else display_text
#                     label = f"{model.names[cls_id]}: {display_text_short}"
#                 else:
#                     label = f"{model.names[cls_id]}: (no text)"
#             except Exception as e:
#                 print(f"Error processing ROI: {e}")
#                 label = f"{model.names[cls_id]}: (processing error)"
                
#             labels.append(label)

#     # Ensure labels match detections
#     if len(labels) != len(detections):
#         # If we have fewer labels than detections, add empty labels
#         while len(labels) < len(detections):
#             labels.append("No text")
#         # If we have more labels than detections, truncate
#         labels = labels[:len(detections)]

#     # Draw boxes and labels
#     if len(detections) > 0:
#         annotated_frame = box_annotator.annotate(annotated_frame, detections)
#         if labels:
#             annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
    
#     # Display current language and controls
#     language_info = f"Language: {languages[current_language]['name']} | TTS: {'ON' if speak_text else 'OFF'}"
#     controls_info = "Press 'l' to change language, 's' to toggle speech, 'r' to read text, 'q' to quit"
    
#     cv2.putText(annotated_frame, language_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     cv2.putText(annotated_frame, controls_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
#     cv2.imshow("YOLO + OCR + Translation", annotated_frame)
    
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
#     elif key == ord('l'):
#         current_language = (current_language + 1) % len(languages)
#         print(f"Language changed to: {languages[current_language]['name']}")
#     elif key == ord('s'):
#         speak_text = not speak_text
#         print(f"Text-to-speech: {'ON' if speak_text else 'OFF'}")
#     elif key == ord('r') and extracted_texts:
#         print("Reading text...")
#         if extracted_texts:
#             combined_text = " ".join(extracted_texts)
#             if current_language > 0:
#                 try:
#                     translated_text = translate_text(combined_text, languages[current_language]["code"])
#                     tts_engine.say(translated_text)
#                     tts_engine.runAndWait()
#                 except Exception as e:
#                     print(f"Error in text-to-speech: {e}")
#             else:
#                 tts_engine.say(combined_text)
#                 tts_engine.runAndWait()

# cap.release()
# cv2.destroyAllWindows()

import cv2
import torch
import pytesseract
import supervision as sv
from ultralytics import YOLO
import pyttsx3
import requests
import time

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# 1. Load YOLO model
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
model = YOLO('yolov10s-doclaynet.pt')

# 2. Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 3. Supervision annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Translation language options
languages = {
    0: {"code": "en", "name": "Original"},
    1: {"code": "hi", "name": "Hindi"},
    2: {"code": "pa", "name": "Punjabi"}
}
current_language = 0  # Default to original (no translation)
speak_text = False  # Flag to control text-to-speech

# Simple translation function using LibreTranslate API
def translate_text(text, target_language):
    if not text or current_language == 0:
        return text
    
    try:
        # Using LibreTranslate public API
        url = "https://translate.terraprint.co/translate"
        
        payload = {
            "q": text,
            "source": "auto",
            "target": target_language,
            "format": "text"
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        result = response.json()
        
        if "translatedText" in result:
            return result["translatedText"]
        else:
            print(f"Translation error: {result}")
            return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# Function to scale detections
def scale_detections(detections, original_shape, new_shape):
    scale_x = new_shape[1] / original_shape[1]
    scale_y = new_shape[0] / original_shape[0]
    
    scaled_xyxy = detections.xyxy.copy()
    scaled_xyxy[:, [0, 2]] *= scale_x
    scaled_xyxy[:, [1, 3]] *= scale_y
    
    return sv.Detections(
        xyxy=scaled_xyxy,
        confidence=detections.confidence,
        class_id=detections.class_id
    )

# Process every n frames to reduce lag
process_every_n_frames = 10
frame_count = 0
last_ocr_time = time.time()
ocr_cooldown = 1.0  # seconds between OCR processing

# Store the last processed results
last_detections = None
last_labels = []
last_extracted_texts = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    current_time = time.time()
    
    # Create a copy for display
    display_frame = frame.copy()
    display_frame = cv2.resize(display_frame, (640, 480))
    
    # Only process frames periodically to reduce lag
    process_this_frame = (frame_count % process_every_n_frames == 0) and (current_time - last_ocr_time >= ocr_cooldown)
    
    if process_this_frame:
        last_ocr_time = current_time
        
        # Resize frame for faster processing
        processed_frame = cv2.resize(frame, (640, 480))
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Run YOLO detection
        results = model(
            processed_frame_rgb,
            imgsz=640,
            conf=0.3,
            iou=0.5
        )[0]

        # Get detections
        last_detections = sv.Detections.from_ultralytics(results)
        last_labels = []
        last_extracted_texts = []

        # Process each detection
        if len(last_detections) > 0:
            for i, (xyxy, conf, cls_id) in enumerate(zip(last_detections.xyxy, last_detections.confidence, last_detections.class_id)):
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Ensure coordinates are within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(processed_frame.shape[1], x2), min(processed_frame.shape[0], y2)
                
                # Skip if region is too small
                if x2 <= x1 or y2 <= y1 or (x2-x1)*(y2-y1) < 100:
                    last_labels.append(f"{model.names[cls_id]}: (region too small)")
                    continue
                    
                roi = processed_frame[y1:y2, x1:x2]

                # Preprocess ROI for better OCR
                try:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Run OCR on the ROI
                    extracted_text = pytesseract.image_to_string(thresh_roi, config='--psm 6')
                    
                    # Clean up text
                    extracted_text = extracted_text.strip().replace("\n", " ")
                    
                    if extracted_text:
                        last_extracted_texts.append(extracted_text)
                        
                        # Translate text if a language is selected
                        if current_language > 0:
                            display_text = translate_text(extracted_text, languages[current_language]["code"])
                        else:
                            display_text = extracted_text
                            
                        # Truncate long text for display
                        display_text_short = display_text[:30] + "..." if len(display_text) > 30 else display_text
                        label = f"{model.names[cls_id]}: {display_text_short}"
                    else:
                        label = f"{model.names[cls_id]}: (no text)"
                except Exception as e:
                    print(f"Error processing ROI: {e}")
                    label = f"{model.names[cls_id]}: (processing error)"
                    
                last_labels.append(label)

            # Ensure labels match detections
            if len(last_labels) != len(last_detections):
                # If we have fewer labels than detections, add empty labels
                while len(last_labels) < len(last_detections):
                    last_labels.append("No text")
                # If we have more labels than detections, truncate
                last_labels = last_labels[:len(last_detections)]

    # Draw boxes and labels on the display frame if we have detections
    if last_detections is not None and len(last_detections) > 0:
        # Scale detections to match the display frame size
        scaled_detections = scale_detections(last_detections, (480, 640), display_frame.shape)
        display_frame = box_annotator.annotate(display_frame, scaled_detections)
        if last_labels:
            display_frame = label_annotator.annotate(display_frame, scaled_detections, last_labels)
    
    # Display current language and controls
    language_info = f"Language: {languages[current_language]['name']} | TTS: {'ON' if speak_text else 'OFF'}"
    controls_info = "Press 'l' to change language, 's' to toggle speech, 'r' to read text, 'q' to quit"
    
    cv2.putText(display_frame, language_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, controls_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow("YOLO + OCR + Translation", display_frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('l'):
        current_language = (current_language + 1) % len(languages)
        print(f"Language changed to: {languages[current_language]['name']}")
    elif key == ord('s'):
        speak_text = not speak_text
        print(f"Text-to-speech: {'ON' if speak_text else 'OFF'}")
    elif key == ord('r') and last_extracted_texts:
        print("Reading text...")
        if last_extracted_texts:
            combined_text = " ".join(last_extracted_texts)
            if current_language > 0:
                try:
                    translated_text = translate_text(combined_text, languages[current_language]["code"])
                    tts_engine.say(translated_text)
                    tts_engine.runAndWait()
                except Exception as e:
                    print(f"Error in text-to-speech: {e}")
            else:
                tts_engine.say(combined_text)
                tts_engine.runAndWait()

cap.release()
cv2.destroyAllWindows()
