import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import cv2 as cv
import numpy as np
import time
import math

just_hand = r"C:\Users\samue\OneDrive\Pictures\Camera Roll\WIN_20260506_17_22_40_Pro.mp4"
sword_sedia = r"C:\Users\samue\Downloads\WIN_20260506_17_13_08_Pro - Trim.mp4"
long_sword_sedia = r"C:\Users\samue\OneDrive\Pictures\Camera Roll\WIN_20260506_17_13_08_Pro.mp4"
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
legRaised90 = False
aspect_ratio = None
feedback = []


def landmarkDistance(landmark1, landmark2):
    return math.sqrt((landmark1[0] - landmark2[0]) ** 2 + (landmark1[1] - landmark2[1]) ** 2+ (landmark1[2] - landmark2[2]) ** 2)
def senangDiriCheck(landmarks):
    left_heel = [landmarks[30].x*aspect_ratio, landmarks[30].y,landmarks[30].z,landmarks[30].visibility]
    right_heel = [landmarks[29].x*aspect_ratio, landmarks[29].y,landmarks[29].z,landmarks[29].visibility]
    
    if left_heel[3] < 0.5 or right_heel[3] < 0.5:
        return False, "Please make sure both your feet are visible for senang diri"
    
    if landmarkDistance(left_heel, right_heel) > 0.33:
        return True,"Legs spread enough"
    else:
        return False, "Legs too close"


def legRaisedAttempt(landmarks):
    left_ankle = [landmarks[28].x*aspect_ratio, landmarks[28].y,landmarks[28].z,landmarks[28].visibility]
    right_ankle = [landmarks[27].x*aspect_ratio, landmarks[27].y,landmarks[27].z,landmarks[27].visibility]

    if right_ankle[1]-left_ankle[1] > 0.04: 
        return True
    else:
        return False

def sediaCheck(landmarks):
    left_heel = [landmarks[30].x*aspect_ratio, landmarks[30].y,landmarks[30].z,landmarks[30].visibility]
    right_heel = [landmarks[29].x*aspect_ratio, landmarks[29].y,landmarks[29].z,landmarks[29].visibility]
    if landmarkDistance(left_heel, right_heel) < 0.33:
        return True,"Legs close enough"
    else:
        return False, "Please bring your legs together for sedia"


def legRaised90Check(landmarks):
    left_hip = [landmarks[24].x*aspect_ratio, landmarks[24].y]
    left_knee = [landmarks[26].x*aspect_ratio, landmarks[26].y]
    return abs(left_hip[1] - left_knee[1]) < 0.05

def swordStraightCheck1(landmarks, hand_landmarks = None):
    left_elbow = [landmarks[13].x*aspect_ratio, landmarks[13].y]
    left_wrist = [landmarks[15].x*aspect_ratio, landmarks[15].y]
    
    isCorrect = 0.055 > left_wrist[1] - left_elbow[1] > 0.040
    feedback = "sword straight" if isCorrect else "sword not straight"
    return feedback

def swordStraightCheck2(landmarks, hand_landmarks = None):
    left_shoulder = [landmarks[11].x * aspect_ratio, landmarks[11].y, landmarks[11].z]
    left_elbow    = [landmarks[13].x * aspect_ratio, landmarks[13].y, landmarks[13].z]
    left_wrist    = [landmarks[15].x * aspect_ratio, landmarks[15].y, landmarks[15].z]
    elbow_to_shoulder = np.array(left_shoulder) - np.array(left_elbow)
    elbow_to_wrist = np.array(left_wrist) - np.array(left_elbow)
    cosine_angle = np.dot(elbow_to_shoulder, elbow_to_wrist) / (np.linalg.norm(elbow_to_shoulder) * np.linalg.norm(elbow_to_wrist))
    angle = np.arccos(cosine_angle) * 180 / np.pi
    return 60 < angle < 120

def forearmStraightCheck(landmarks, hand_landmarks = None):
    left_elbow = [landmarks[13].x*aspect_ratio, landmarks[13].y, landmarks[13].z]
    left_wrist = [landmarks[15].x*aspect_ratio, landmarks[15].y, landmarks[15].z]
    
    return 0.03 > left_wrist[1]-left_elbow[1] > 0

def main(pose_result, hand_result):
    global feedback, state, legRaised90, aspect_ratio
    if not pose_result or not pose_result.pose_landmarks:
        return

    landmarks = pose_result.pose_landmarks[0]
    feedback = []


    if state == "start":
        isCorrect, temp_feedback = senangDiriCheck(landmarks)
        if isCorrect:
            state = "senang_diri"
        
        feedback.append(temp_feedback)


    elif state == "senang_diri":
        isCorrect, temp_feedback = senangDiriCheck(landmarks)
        if not isCorrect:
            state = "start"
        else:
            if legRaisedAttempt(landmarks):
                state = "leg_raised_attempt"
        feedback.append(temp_feedback)

    elif state == "leg_raised_attempt":
        isCorrect = legRaisedAttempt(landmarks)
        if legRaised90Check(landmarks):
            legRaised90 = True
        if not isCorrect:
            if legRaised90:
                state = "sedia"
                legRaised90 = False
            else:
                state = "senang_diri"

    elif state == "sedia":
        isCorrect, temp_feedback = sediaCheck(landmarks)
        
        feedback.append(temp_feedback)



    if forearmStraightCheck(landmarks):
        feedback.append("forearm straight")
    else:
        feedback.append("forearm not straight")
    

            


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

def drawSkeleton(image1, pose_result, hand_result):
    new_image = np.copy(image1)
    if pose_result and pose_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image=new_image,
            landmark_list=pose_result.pose_landmarks[0],
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2))
    if hand_result and hand_result.hand_landmarks:
        drawing_utils.draw_landmarks(
            image=new_image,
            landmark_list=hand_result.hand_landmarks[0],
            connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=drawing_styles.get_default_hand_connections_style())
    return new_image

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

        height, width, _ = frame.shape
        aspect_ratio = width / height

        main(pose_result, hand_result)

        font_scale = width / 1000
        feedback.append(f"state: {state}")
        for i, line in enumerate(feedback):
            (text_width, text_height), _ = cv.getTextSize(line, cv.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            x = (width - text_width) // 2
            y = int(height * 0.1) + i * int(text_height * 1.5)
            cv.putText(frame, line, (x, y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)

        feedback = []
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        annotated_frame = drawSkeleton(frame, pose_result, hand_result)
        display = cv.resize(annotated_frame, (1320, 680))
        cv.imshow("Drill Checker", display)

        if cv.waitKey(int(1)) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()