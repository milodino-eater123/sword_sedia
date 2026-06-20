import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
from carry_sword_45deg import main
from helper_functions import drawSkeleton, draw_feedback

VIDEO_PATH = r"C:\Users\samue\Downloads\WIN_20260506_17_13_08_Pro - Trim.mp4"
POSE_MODEL_PATH = r"C:\Users\samue\Downloads\pose_landmarker_full.task"
HAND_MODEL_PATH = r"C:\Users\samue\Downloads\hand_landmarker.task"
DISPLAY_SIZE = (1320, 680)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
)
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.20,
)

cap = cv.VideoCapture(VIDEO_PATH)
fps = cap.get(cv.CAP_PROP_FPS) or 30
ms_per_frame = 1000 / fps

state = "start"
legRaised90 = False

with PoseLandmarker.create_from_options(options) as landmarker, \
     HandLandmarker.create_from_options(hand_options) as hand_landmarker:
    frame_idx = 0
    while True:
        working, frame = cap.read()
        if not working:
            break

        frame = cv.flip(frame, 1)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        timestamp_ms = int(frame_idx * ms_per_frame)
        frame_idx += 1

        pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        height, width, _ = frame.shape
        aspect_ratio = width / height

        state, legRaised90, feedback = main(pose_result, hand_result, aspect_ratio, state, legRaised90)

        feedback.append(f"state: {state}")
        draw_feedback(frame, feedback)

        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        annotated_frame = drawSkeleton(frame, pose_result, hand_result)
        display = cv.resize(annotated_frame, DISPLAY_SIZE)
        cv.imshow("Drill Checker", display)

        if cv.waitKey(1) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
