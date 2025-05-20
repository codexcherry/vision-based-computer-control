import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

# Prevent mouse from going to screen corners
pyautogui.FAILSAFE = False
# Smooth mouse movement
pyautogui.MINIMUM_DURATION = 0
pyautogui.PAUSE = 0.01

# Initialize MediaPipe Hand and Face tracking
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Initialize smoothing variables
smoothing = 0.5
prev_x, prev_y = screen_width/2, screen_height/2

# Initialize gesture tracking
scroll_cooldown = 0
SCROLL_COOLDOWN_TIME = 0.2  # Increased cooldown for better control
SCROLL_AMOUNT = 50  # Scroll amount per action

# Initialize eye blink detection
BLINK_THRESHOLD = 0.25
last_blink_time = time.time()
MIN_BLINK_INTERVAL = 0.3
last_ear = 1.0

# Hand height range for click detection
HAND_HEIGHT_MIN = 0.2  # Minimum height for click detection
HAND_HEIGHT_MAX = 0.6  # Maximum height for click detection

def check_fingers_up(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    finger_bases = [5, 9, 13, 17]
    count = 0
    
    # Check thumb separately
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        count += 1
    
    # Check other fingers
    for tip, base in zip(finger_tips, finger_bases):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            count += 1
            
    return count

def calculate_eye_aspect_ratio(landmarks, left_eye, right_eye):
    def get_distance(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    left_eye_ratio = (get_distance(landmarks[left_eye[1]], landmarks[left_eye[5]]) + 
                     get_distance(landmarks[left_eye[2]], landmarks[left_eye[4]])) / (2 * get_distance(landmarks[left_eye[0]], landmarks[left_eye[3]]))
    
    right_eye_ratio = (get_distance(landmarks[right_eye[1]], landmarks[right_eye[5]]) + 
                      get_distance(landmarks[right_eye[2]], landmarks[right_eye[4]])) / (2 * get_distance(landmarks[right_eye[0]], landmarks[right_eye[3]]))
    
    return (left_eye_ratio + right_eye_ratio) / 2

def calculate_volume_from_hand(hand_landmarks):
    index_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]
    
    distance = np.sqrt(
        (index_tip.x - thumb_tip.x)**2 + 
        (index_tip.y - thumb_tip.y)**2 + 
        (index_tip.z - thumb_tip.z)**2
    )
    
    return distance

def is_hand_in_click_zone(hand_landmarks):
    # Check if hand is in the right height range for clicking
    index_tip_y = hand_landmarks.landmark[8].y
    return HAND_HEIGHT_MIN <= index_tip_y <= HAND_HEIGHT_MAX

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    hand_output = hands.process(rgb_frame)
    face_output = face_mesh.process(rgb_frame)
    
    frame_height, frame_width, _ = frame.shape
    
    # Variables to track right hand state
    right_hand_present = False
    right_hand_in_zone = False

    # Handle hand gestures first
    if hand_output.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_output.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if hand_output.multi_handedness:
                handedness = hand_output.multi_handedness[idx].classification[0].label
                
                if handedness == "Right":
                    right_hand_present = True
                    right_hand_in_zone = is_hand_in_click_zone(hand_landmarks)
                    
                    # Get index finger tip coordinates
                    index_finger = hand_landmarks.landmark[8]
                    
                    # Convert coordinates to screen position
                    screen_x = screen_width * 1.5 * (index_finger.x - 0.5) + screen_width/2
                    screen_y = screen_height * 1.5 * (index_finger.y - 0.5) + screen_height/2
                    
                    screen_x = np.clip(screen_x, 0, screen_width)
                    screen_y = np.clip(screen_y, 0, screen_height)
                    
                    smooth_x = prev_x * smoothing + screen_x * (1 - smoothing)
                    smooth_y = prev_y * smoothing + screen_y * (1 - smoothing)
                    
                    pyautogui.moveTo(smooth_x, smooth_y)
                    prev_x, prev_y = smooth_x, smooth_y
                    
                    # Check fingers for scrolling
                    fingers_up = check_fingers_up(hand_landmarks)
                    current_time = time.time()
                    
                    if current_time - scroll_cooldown > SCROLL_COOLDOWN_TIME:
                        if fingers_up >= 4:  # All fingers up - scroll up
                            pyautogui.scroll(SCROLL_AMOUNT)
                            print("scroll up")
                            scroll_cooldown = current_time
                        elif fingers_up <= 1:  # All fingers down - scroll down
                            pyautogui.scroll(-SCROLL_AMOUNT)
                            print("scroll down")
                            scroll_cooldown = current_time
                    
                    # Draw click zone indicator
                    zone_y_min = int(HAND_HEIGHT_MIN * frame_height)
                    zone_y_max = int(HAND_HEIGHT_MAX * frame_height)
                    cv2.rectangle(frame, (frame_width-50, zone_y_min), (frame_width-20, zone_y_max), 
                                (0, 255, 0) if right_hand_in_zone else (0, 0, 255), 2)
                
                elif handedness == "Left":  # Left hand controls volume
                    fingers_up = check_fingers_up(hand_landmarks)
                    
                    if fingers_up in [0, 2]:
                        volume_metric = calculate_volume_from_hand(hand_landmarks)
                        vol = np.interp(volume_metric, [0.03, 0.2], [minVol, maxVol])
                        vol = np.clip(vol, minVol, maxVol)
                        
                        current_vol = volume.GetMasterVolumeLevelScalar()
                        target_vol = np.interp(vol, [minVol, maxVol], [0, 1])
                        smooth_vol = current_vol * 0.5 + target_vol * 0.5
                        
                        volume.SetMasterVolumeLevelScalar(smooth_vol, None)
                        vol_percentage = int(smooth_vol * 100)
                        
                        cv2.rectangle(frame, (50, 150), (85, 400), (255, 0, 0), 3)
                        cv2.rectangle(frame, (50, int(400 - vol_percentage * 2.5)), (85, 400), (255, 0, 0), cv2.FILLED)
                        cv2.putText(frame, f'{vol_percentage}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                        
                        index_y = int(hand_landmarks.landmark[8].y * frame_height)
                        cv2.circle(frame, (67, index_y), 10, (0, 255, 0), cv2.FILLED)
                    else:
                        cv2.putText(frame, "Show 2 fingers for volume control", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Handle eye blink detection only if right hand is present and in the click zone
    if face_output.multi_face_landmarks and right_hand_present and right_hand_in_zone:
        face_landmarks = face_output.multi_face_landmarks[0].landmark
        
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        ear = calculate_eye_aspect_ratio(face_landmarks, LEFT_EYE, RIGHT_EYE)
        
        current_time = time.time()
        if ear < BLINK_THRESHOLD and last_ear > BLINK_THRESHOLD and (current_time - last_blink_time) > MIN_BLINK_INTERVAL:
            pyautogui.click()
            print("click (blink)")
            last_blink_time = current_time
        
        last_ear = ear
        
        # Show eye state only when hand is in click zone
        cv2.putText(frame, f'EAR: {ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 0) if ear > BLINK_THRESHOLD else (0, 0, 255), 2)
    
    cv2.imshow('Hand Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()