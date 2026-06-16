import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import numpy as np
import time
from carry_sword_45deg import main
from helper_functions import drawSkeleton

just_hand = r"C:\Users\samue\OneDrive\Pictures\Camera Roll\WIN_20260506_17_22_40_Pro.mp4"
sword_sedia = r"C:\Users\samue\Downloads\WIN_20260506_17_13_08_Pro - Trim.mp4"
long_sword_sedia = r"C:\PUsers\samue\OneDrive\Pictures\Camera Roll\WIN_20260506_17_13_08_Pro.mp4"
cap = cv.VideoCapture(sword_sedia)  
model_path = r"C:\Users\samue\Downloads\pose_landmarker_full.task"
hand_model_path = r"C:\Users\samue\Downloads\hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

state = "start"
feedback = []



options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
)
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.20,
)


with PoseLandmarker.create_from_options(options) as landmarker, \
     HandLandmarker.create_from_options(hand_options) as hand_landmarker:
    while True:
        working, frame = cap.read()
        if not working:
            break

        frame = cv.flip(frame, 1)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        timestamp_ms = int(time.time() * 1000)

        pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        main()

        #PUT TEXT ON SCREEN
        height, width, _ = frame.shape
        aspect_ratio = width / height
    
        font_scale = width / 1000
        feedback.append(f"state: {state}")
        for i, line in enumerate(feedback):
            (text_width, text_height), _ = cv.getTextSize(line, cv.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            x = (width - text_width) // 2
            y = int(height * 0.1) + i * int(text_height * 1.5)
            cv.putText(frame, line, (x, y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)
        feedback = []
        #PUT TEXT ON SCREEN
        
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        annotated_frame = drawSkeleton(frame, pose_result,hand_result)
        display = cv.resize(annotated_frame, (1320, 680))
        cv.imshow("Drill Checker", display)

        if cv.waitKey(int(1)) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()