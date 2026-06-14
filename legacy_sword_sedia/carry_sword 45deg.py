import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import cv2 as cv
import time
import numpy as np
import math

#boring stuff
cap = cv.VideoCapture(0)
model_path = r"C:\Users\samue\Downloads\pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarkerResult = vision.PoseLandmarkerResult
RunningMode = vision.RunningMode

#intialize variables
latest_result = None
state = "start"
legRaised90 = False
aspect_ratio = None
feedback = []

#MEDIAPIPE HANDS STUFF

HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult

hand_model_path = r"C:\Users\samue\Downloads\hand_landmarker.task" 
latest_hand_result = None

# Hand callback
def handCallbackFunction(result, image, timestamp):
    global latest_hand_result
    latest_hand_result = result

# Hand options
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=handCallbackFunction,
    num_hands=2,
)

def callbackFunction(result,image,timestamp):
    global latest_result
    latest_result = result

def drawSkeleton(image1,result):
    
        
    new_image = np.copy(image1)

    if result and result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        drawing_utils.draw_landmarks(
            image=new_image,
            landmark_list=landmarks, 
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec= drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec= drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2))
    
    if latest_hand_result and latest_hand_result.hand_landmarks:
        hand_landmarks = latest_hand_result.hand_landmarks[0]
        drawing_utils.draw_landmarks(
            image=new_image,
            landmark_list=hand_landmarks,
            connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=drawing_styles.get_default_hand_connections_style())
    return new_image

def landmarkDistance(landmark1, landmark2):
    return math.sqrt((landmark1[0] - landmark2[0]) ** 2 + (landmark1[1] - landmark2[1]) ** 2+ (landmark1[2] - landmark2[2]) ** 2)
def senangDiriCheck(landmarks):
    left_heel = [landmarks[30].x*aspect_ratio, landmarks[30].y,landmarks[30].z,landmarks[30].visibility]
    right_heel = [landmarks[29].x*aspect_ratio, landmarks[29].y,landmarks[29].z,landmarks[29].visibility]
    
    if left_heel[3] < 0.5 or right_heel[3] < 0.5:
        return False, ["Please make sure both your feet are visible for senang diri"]
    
    if landmarkDistance(left_heel, right_heel) > 0.04:
        return True,[]
    else:
        return False, ["Please spread your legs wider for senang diri"]


def legRaisedAttempt(landmarks):
    left_ankle = [landmarks[28].x*aspect_ratio, landmarks[28].y,landmarks[28].z,landmarks[28].visibility]
    right_ankle = [landmarks[27].x*aspect_ratio, landmarks[27].y,landmarks[27].z,landmarks[27].visibility]

    if right_ankle[1]-left_ankle[1] > 0.02: 
        return True
    else:
        return False

def sediaCheck(landmarks):
    left_heel = [landmarks[30].x*aspect_ratio, landmarks[30].y,landmarks[30].z,landmarks[30].visibility]
    right_heel = [landmarks[29].x*aspect_ratio, landmarks[29].y,landmarks[29].z,landmarks[29].visibility]
    if landmarkDistance(left_heel, right_heel) < 0.05:
        return True,[]
    else:
        return False, ["Please bring your legs together for sedia"]

def legRaised90Check(landmarks):
    left_hip = [landmarks[24].x*aspect_ratio, landmarks[24].y]
    left_knee = [landmarks[26].x*aspect_ratio, landmarks[26].y]
    
    return abs(left_hip[1] - left_knee[1]) < 0.02

def swordStraightCheck1(landmarks, hand_landmarks):
    left_elbow = [landmarks[13].x*aspect_ratio, landmarks[13].y]
    middle_knuckle = [hand_landmarks[9].x*aspect_ratio, hand_landmarks[9].y]
    #in the future, can be relative to shoulder-elbow length for diff body types
    
    return abs(left_elbow[1] - middle_knuckle[1]) < 0.10

def swordStraightCheck2(landmarks, hand_landmarks):
    left_shoulder = [landmarks[11].x * aspect_ratio, landmarks[11].y, landmarks[11].z]
    left_elbow    = [landmarks[13].x * aspect_ratio, landmarks[13].y, landmarks[13].z]
    left_wrist    = [landmarks[15].x * aspect_ratio, landmarks[15].y, landmarks[15].z]


    elbow_to_shoulder = np.array(left_shoulder) - np.array(left_elbow)
    elbow_to_wrist = np.array(left_wrist) - np.array(left_elbow)

    cosine_angle = np.dot(elbow_to_shoulder, elbow_to_wrist) / (np.linalg.norm(elbow_to_shoulder) * np.linalg.norm(elbow_to_wrist))
    angle = np.arccos(cosine_angle) * 180 / np.pi

    return 60 < angle < 120


#drill detection logic
def main():
    global feedback, isFormCorrect,latest_result,sword_detected,state,legRaised90,aspect_ratio
    if not latest_result or not latest_result.pose_landmarks:
        return

    landmarks = latest_result.pose_landmarks[0]


    if state == "start":
        isCorrect, temp_feedback = senangDiriCheck(landmarks)
        if isCorrect:
            state = "senang_diri"
        else:
            feedback = temp_feedback

    #if senang diri, check if transition to leg raised attempt
    elif state == "senang_diri":
        isCorrect, temp_feedback = senangDiriCheck(landmarks)
        if not isCorrect:
            feedback = temp_feedback
            state = "start"
        else:
            if legRaisedAttempt(landmarks):
                state = "leg_raised_attempt"
    
    elif state == "leg_raised_attempt":
        isCorrect= legRaisedAttempt(landmarks)
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
        if not isCorrect:
            feedback = temp_feedback
    


            


#main
options = PoseLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=model_path),
    running_mode = RunningMode.LIVE_STREAM,
    result_callback = callbackFunction,
)

with PoseLandmarker.create_from_options(options) as landmarker, HandLandmarker.create_from_options(hand_options) as hand_landmarker:
    while True:
        working,frame = cap.read()
        if not working:
            continue
        frame = cv.flip(frame, 1)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(time.time()*1000)
        
        landmarker.detect_async(mp_image, timestamp)
        hand_landmarker.detect_async(mp_image, timestamp)
        hand_result = latest_hand_result 
        result = latest_result


        height, width, _ = frame.shape
        aspect_ratio = width / height

        main()

        font_scale = width / 1000
        feedback.append(f"state: {state}")
        for i, line in enumerate(feedback):
            (text_width, text_height), _ = cv.getTextSize(line, cv.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            x = (width - text_width) // 2
            y = int(height * 0.1) + i * int(text_height * 1.5)
            cv.putText(frame, line, (x, y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)
        
        feedback = []
        
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        annotated_frame = drawSkeleton(frame,result) 
        if annotated_frame is None:
            annotated_frame = frame

        cv.imshow("Drill Checker", annotated_frame) 
        if cv.waitKey(1) == ord('q'):
            break
    
cap.release()
cv.destroyAllWindows()