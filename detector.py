import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import defaultdict
import time

# --- TTS ---
try:
    import pyttsx3
    TTS_ENABLED = True
    # Initialize the TTS engine ONCE
    tts_engine = pyttsx3.init()
    print("TTS library (pyttsx3) loaded.")

    # --- Find Available Voices (Thai and English) ---
    voices = tts_engine.getProperty('voices')
    thai_voice_id = None
    english_voice_id = None
    print("\nSearching for suitable TTS Voices...")
    print("-" * 30)
    for voice in voices:
        # print(f"ID: {voice.id} | Name: {voice.name} | Langs: {voice.languages}") # Uncomment for full debug
        try:
            # Check for Thai (th_TH, th) - take the first match
            if not thai_voice_id and any(lang.lower().startswith('th') for lang in voice.languages):
                 thai_voice_id = voice.id
                 print(f"* Found Thai Voice: {voice.name} (ID: {voice.id})")

            # Check for English (en_US, en_GB, en) - prioritize US/GB, take first match
            if not english_voice_id and any(lang.lower().startswith('en') for lang in voice.languages):
                 english_voice_id = voice.id
                 print(f"* Found English Voice: {voice.name} (ID: {voice.id})")

            # Stop if both found
            if thai_voice_id and english_voice_id:
                break
        except Exception as e:
             print(f"Warning: Could not process voice {voice.id}. Error: {e}")

    print("-" * 30)

    if not thai_voice_id:
        print("WARNING: No specific Thai voice found. Thai TTS may fail or use default.")
    if not english_voice_id:
        print("WARNING: No specific English voice found. English TTS may fail or use default.")
        # Fallback: Use the default voice ID if no English one is explicitly found
        if voices: english_voice_id = voices[0].id # Use the very first voice as a fallback

    # Set default properties (optional)
    # tts_engine.setProperty('rate', 150)
    # tts_engine.setProperty('volume', 0.9)

except ImportError:
    print("WARNING: pyttsx3 not found. TTS features will be disabled.")
    TTS_ENABLED = False
    tts_engine = None
    thai_voice_id = None
    english_voice_id = None

# --- Configuration ---
MODEL_PATH = 'yolov8n.pt' # <-- Or your 'trash_model.pt'
CONFIDENCE_THRESHOLD_DISPLAY = 0.45
CONFIDENCE_THRESHOLD_SORT = 0.60
VIDEO_SOURCE = 0

# --- Modified Trash Class to Bin Mapping (Simplified to 3 bins) ---
TRASH_TO_BIN_MAP = { # Keys should be lowercase for reliable matching
    # Recycle items
    'bottle': 'Recycle', 'wine glass': 'Recycle', 'book': 'Recycle',
    'cup': 'Recycle', 'bowl': 'Recycle', 'cell phone': 'Recycle',
    'remote': 'Recycle', 'tv': 'Recycle', 'laptop': 'Recycle',
    'mouse': 'Recycle', 'keyboard': 'Recycle',
    
    # Food items
    'banana': 'Food', 'apple': 'Food', 'orange': 'Food',
    'broccoli': 'Food', 'carrot': 'Food', 'pizza': 'Food',
    'sandwich': 'Food', 'hot dog': 'Food', 'cake': 'Food',
    
    # Hazardous items
    'toothbrush': 'Hazardous', 'fork': 'Hazardous', 'knife': 'Hazardous',
    'spoon': 'Hazardous', 'scissors': 'Hazardous', 'battery': 'Hazardous',
    
    # Add more items as needed
}

VALID_SORTING_BINS = {'Recycle', 'Food', 'Hazardous'}
AMBIGUOUS_CLASSES = {'cup', 'bowl', 'container', 'plate', 'wrapper', 'bag'}
DEFAULT_BIN = 'Recycle'  # Changed default bin

# --- Visual Configuration ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6; TEXT_THICKNESS = 1; BOX_THICKNESS = 3
COLORS = { # BGR
    'Recycle': (255, 150, 0),     # Blue with orange tint
    'Food': (0, 165, 255),        # Orange with yellow tint
    'Hazardous': (0, 0, 255),     # Red
    'Uncertain': (0, 255, 255),   # Yellow
    'Unknown': (255, 0, 255),     # Purple
    'Default': (0, 200, 0)        # Green
}
INFO_TEXT_COLOR = (255, 255, 255); INFO_BG_COLOR = (50, 50, 50); INFO_PANEL_ALPHA = 0.7

# --- TTS and Sensor Simulation Configuration ---
TTS_COOLDOWN_SECONDS = 3.0  # Longer cooldown to prevent speech overlap
THAI_BIN_NAMES = { # Thai phrases (modified) - kept for compatibility
    'Recycle': 'ถังรีไซเคิล', 
    'Food': 'ถังอาหาร',
    'Hazardous': 'ถังขยะอันตราย',
}
ENGLISH_BIN_NAMES = { # English phrases (modified) - kept for compatibility
    'Recycle': 'Put in Recycle Bin', 
    'Food': 'Put in Food Waste Bin',
    'Hazardous': 'Put in Hazardous Waste Bin',
}

# Voice commands for each bin - these will be spoken aloud
SOUND_COMMANDS = {
    'Recycle': "Please place in the recycling bin",
    'Food': "This is food waste. Place in the food bin",
    'Hazardous': "Warning! Hazardous material. Use the hazardous waste bin",
}

# State variables
last_announced_bin = None
last_announced_time = 0
current_best_object_info = None

# --- Helper Function: Draw Text ---
def draw_text_with_background(frame, text, position, font, scale, text_color, bg_color, thickness, padding=2):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    text_height += baseline
    bg_top_left = position
    bg_bottom_right = (position[0] + text_width + padding * 2, position[1] + text_height + padding * 2)
    text_origin = (position[0] + padding, position[1] + text_height - baseline // 2 + padding)
    frame_h, frame_w = frame.shape[:2]
    if bg_top_left[0] < 0: bg_top_left = (0, bg_top_left[1])
    if bg_top_left[1] < 0: bg_top_left = (bg_top_left[0], 0)
    if bg_bottom_right[0] > frame_w: bg_bottom_right = (frame_w, bg_bottom_right[1])
    if bg_bottom_right[1] > frame_h: bg_bottom_right = (bg_bottom_right[0], frame_h)
    text_origin = (bg_top_left[0] + padding, bg_top_left[1] + text_height - baseline // 2 + padding)
    if text_origin[1] > frame_h - baseline//2: text_origin = (text_origin[0], frame_h - baseline//2)
    cv2.rectangle(frame, bg_top_left, bg_bottom_right, bg_color, -1)
    cv2.putText(frame, text, text_origin, font, scale, text_color, thickness, cv2.LINE_AA)
    return bg_bottom_right[1] - bg_top_left[1]


# --- TTS Function (enhanced for reliable audio output) ---
def speak_text(text, language='en'):
    """Uses pyttsx3 engine to generate actual speech for the given text."""
    global tts_engine, thai_voice_id, english_voice_id
    if not TTS_ENABLED or not text or not tts_engine:
        print(f"TTS not available: Would have spoken '{text}'")
        return

    target_voice_id = None
    if language == 'th':
        target_voice_id = thai_voice_id
        lang_name = "Thai"
    else: # Default to English
        target_voice_id = english_voice_id
        lang_name = "English"

    if not target_voice_id:
        print(f"TTS Warning: No voice ID found for {lang_name}. Using default voice for '{text}'.")
        # Continue with default voice rather than returning

    try:
        # Adjust rate and volume for better clarity
        tts_engine.setProperty('rate', 150)  # Speed - words per minute
        tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        
        # Set voice if available
        if target_voice_id:
            current_voice = tts_engine.getProperty('voice')
            if current_voice != target_voice_id:
                print(f"TTS: Setting voice to {lang_name}")
                tts_engine.setProperty('voice', target_voice_id)

        print(f"TTS: Generating speech for '{text}'")
        tts_engine.say(text)
        tts_engine.runAndWait()  # This line is critical - it blocks until speech is complete
        print(f"TTS: Finished speaking")

    except Exception as e:
        print(f"Error during speech generation: {e}")


# --- Sound Command Function ---
def play_bin_sound_command(bin_name):
    """Play a specific sound command for the detected bin using text-to-speech"""
    if bin_name in SOUND_COMMANDS:
        command_text = SOUND_COMMANDS[bin_name]
        print(f"SOUND COMMAND: Playing audio for '{command_text}'")
        # Use the speak_text function to actually speak the command through pyttsx3
        speak_text(command_text, 'en')  # Always use English for commands
    else:
        print(f"No sound command defined for bin: {bin_name}")


# --- Simulated Sensor Activation ---
def activate_bin_sensor(bin_name):
    if bin_name in VALID_SORTING_BINS:
        print(f"--- SENSOR SIM: Activating **{bin_name.upper()}** bin ---")
    else:
        print(f"--- SENSOR SIM: No action for bin '{bin_name}' ---")

# --- Load Model ---
try:
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    if hasattr(model, 'names'): class_names = model.names; print("Model loaded.")
    else: print("Warning: Cannot get class names."); class_names = {}
except Exception as e: print(f"Error loading model: {e}"); exit()

# --- Initialize Video ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened(): print(f"Error opening video source {VIDEO_SOURCE}"); exit()
print(f"Starting video stream. Sort threshold={CONFIDENCE_THRESHOLD_SORT:.2f}")
print("Press 'q' to quit.")

# --- Initialize TTS with a test message ---
if TTS_ENABLED and tts_engine:
    print("\nTesting text-to-speech functionality...")
    try:
        speak_text("Trash sorting system initialized. Ready to detect objects.", 'en')
        print("TTS test successful. Audio output is working.\n")
    except Exception as e:
        print(f"TTS test failed: {e}")
        print("The system will continue without audio feedback.\n")
        TTS_ENABLED = False

# --- Main Loop ---
while True:
    success, frame = cap.read()
    if not success: print("End of stream."); break
    frame_height, frame_width = frame.shape[:2]

    best_detection_in_frame = None
    highest_confidence = 0.0

    # --- Perform Detection ---
    results = model(frame, stream=False, conf=CONFIDENCE_THRESHOLD_DISPLAY, verbose=False)

    # --- Find the single best "trash" object ---
    if results and results[0].boxes:
        for box in results[0].boxes:
            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD_SORT: continue

            cls_id = int(box.cls[0])
            class_name = class_names.get(cls_id, 'Unknown Class')

            # Skip persons and unknown classes
            if class_name.lower() == 'person' or class_name == 'Unknown Class': continue

            lower_class_name = class_name.lower()
            target_bin = TRASH_TO_BIN_MAP.get(lower_class_name, DEFAULT_BIN)

            # Only consider objects mapping to valid sorting bins
            if target_bin in VALID_SORTING_BINS:
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_detection_in_frame = {
                        'box': list(map(int, box.xyxy[0])), 'confidence': confidence,
                        'class_name': class_name, 'target_bin': target_bin,
                        'is_ambiguous': lower_class_name in AMBIGUOUS_CLASSES
                    }

    # --- Process the single best detection ---
    current_best_object_info = None
    if best_detection_in_frame:
        current_best_object_info = best_detection_in_frame
        b_info = best_detection_in_frame
        target_bin = b_info['target_bin'] # English key, e.g., 'Recycle'

        # --- TTS and Sensor Trigger Logic ---
        now = time.time()
        if target_bin != last_announced_bin or (now - last_announced_time > TTS_COOLDOWN_SECONDS):
            if TTS_ENABLED and tts_engine:
                # Play the actual sound command using pyttsx3
                play_bin_sound_command(target_bin)
            else:
                print(f"TTS DISABLED: Would announce '{SOUND_COMMANDS.get(target_bin, f'Place in {target_bin} bin')}'")
            
            # Activate sensor after speaking
            activate_bin_sensor(target_bin)

            # Update state
            last_announced_bin = target_bin
            last_announced_time = now

        # --- Draw Visuals for the best object ONLY ---
        x1, y1, x2, y2 = b_info['box']
        confidence = b_info['confidence']
        class_name = b_info['class_name']
        color = COLORS.get(target_bin, COLORS['Default'])
        bin_label_prefix = "Bin: "
        if b_info['is_ambiguous']:
             bin_label_prefix = "Bin (Check?): "
        detailed_label = f"{class_name} ({confidence:.2f})"
        final_bin_label = f"{bin_label_prefix}{target_bin}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS) # Draw thick box

        # Draw labels (above/below logic)
        (tw1, th1), bl1 = cv2.getTextSize(detailed_label, FONT, FONT_SCALE, TEXT_THICKNESS); tbh1 = th1 + bl1 + 4
        (tw2, th2), bl2 = cv2.getTextSize(final_bin_label, FONT, FONT_SCALE, TEXT_THICKNESS); tbh2 = th2 + bl2 + 4
        total_label_h = tbh1 + tbh2
        label_y_above = y1 - total_label_h - 5
        label_y_below = y2 + 5
        if label_y_above < 0: # Draw below
            curr_y = label_y_below
            h1 = draw_text_with_background(frame, detailed_label, (x1, curr_y), FONT, FONT_SCALE, INFO_TEXT_COLOR, color, TEXT_THICKNESS)
            curr_y += h1
            draw_text_with_background(frame, final_bin_label, (x1, curr_y), FONT, FONT_SCALE, INFO_TEXT_COLOR, color, TEXT_THICKNESS)
        else: # Draw above
            curr_y = label_y_above
            h1 = draw_text_with_background(frame, detailed_label, (x1, curr_y), FONT, FONT_SCALE, INFO_TEXT_COLOR, color, TEXT_THICKNESS)
            curr_y += h1
            draw_text_with_background(frame, final_bin_label, (x1, curr_y), FONT, FONT_SCALE, INFO_TEXT_COLOR, color, TEXT_THICKNESS)

    else: # No best object found this frame
        if time.time() - last_announced_time > TTS_COOLDOWN_SECONDS * 1.5:
             last_announced_bin = None


    # --- Information Overlay Panel ---
    panel_x, panel_y, panel_w, panel_h = 10, 10, 320, 155  # Slightly taller panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), INFO_BG_COLOR, -1)
    cv2.addWeighted(overlay, INFO_PANEL_ALPHA, frame, 1 - INFO_PANEL_ALPHA, 0, frame)
    line_h = 28; curr_y = panel_y + line_h - 10
    
    # Title
    cv2.putText(frame, "Trash Sorting System", (panel_x + 10, curr_y), FONT, FONT_SCALE, INFO_TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)
    curr_y += line_h
    
    # Confidence threshold
    cv2.putText(frame, f"Sort Conf: {CONFIDENCE_THRESHOLD_SORT:.2f}", (panel_x + 10, curr_y), FONT, FONT_SCALE, INFO_TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)
    curr_y += line_h
    
    # Current status
    status_text = "Status: Searching..."
    status_color = INFO_TEXT_COLOR
    if current_best_object_info:
        b_info = current_best_object_info
        status_text = f"Target: {b_info['class_name']} -> {b_info['target_bin']}"
        status_color = COLORS.get(b_info['target_bin'], COLORS['Default'])
    cv2.putText(frame, status_text, (panel_x + 10, curr_y), FONT, FONT_SCALE, status_color, TEXT_THICKNESS, cv2.LINE_AA)
    curr_y += line_h
    
    # Last action
    last_bin_text = f"Last Action: {last_announced_bin or 'None'}"
    cv2.putText(frame, last_bin_text, (panel_x + 10, curr_y), FONT, FONT_SCALE, INFO_TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)
    curr_y += line_h
    
    # Audio status
    audio_status = "Audio: ENABLED" if TTS_ENABLED else "Audio: DISABLED"
    cv2.putText(frame, audio_status, (panel_x + 10, curr_y), FONT, FONT_SCALE, INFO_TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)


    # --- Display ---
    cv2.imshow("Trash Sorting with Sound Commands", frame)

    # --- Exit ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): print("Exiting..."); break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Resources released.")