import numpy as np
import math


def landmarkDistance(landmark1, landmark2):
    return math.sqrt((landmark1[0] - landmark2[0]) ** 2 + (landmark1[1] - landmark2[1]) ** 2+ (landmark1[2] - landmark2[2]) ** 2)
def senangDiriCheck(landmarks, aspect_ratio):
    left_heel = [landmarks[30].x*aspect_ratio, landmarks[30].y,landmarks[30].z,landmarks[30].visibility]
    right_heel = [landmarks[29].x*aspect_ratio, landmarks[29].y,landmarks[29].z,landmarks[29].visibility]
    
    if left_heel[3] < 0.5 or right_heel[3] < 0.5:
        return False, ["Please make sure both your feet are visible for senang diri"]
    
    if landmarkDistance(left_heel, right_heel) > 0.04:
        return True,[]
    else:
        return False, ["Please spread your legs wider for senang diri"]


def legRaisedAttempt(landmarks, aspect_ratio):
    left_ankle = [landmarks[28].x*aspect_ratio, landmarks[28].y,landmarks[28].z,landmarks[28].visibility]
    right_ankle = [landmarks[27].x*aspect_ratio, landmarks[27].y,landmarks[27].z,landmarks[27].visibility]

    if right_ankle[1]-left_ankle[1] > 0.02: 
        return True
    else:
        return False

def sediaCheck(landmarks, aspect_ratio):
    left_heel = [landmarks[30].x*aspect_ratio, landmarks[30].y,landmarks[30].z,landmarks[30].visibility]
    right_heel = [landmarks[29].x*aspect_ratio, landmarks[29].y,landmarks[29].z,landmarks[29].visibility]
    if landmarkDistance(left_heel, right_heel) < 0.05:
        return True,[]
    else:
        return False, ["Please bring your legs together for sedia"]

def legRaised90Check(landmarks, aspect_ratio):
    left_hip = [landmarks[24].x*aspect_ratio, landmarks[24].y]
    left_knee = [landmarks[26].x*aspect_ratio, landmarks[26].y]
    
    return abs(left_hip[1] - left_knee[1]) < 0.02

def swordStraightCheck1(landmarks, hand_landmarks, aspect_ratio):
    left_elbow = [landmarks[13].x*aspect_ratio, landmarks[13].y]
    middle_knuckle = [hand_landmarks[9].x*aspect_ratio, hand_landmarks[9].y]
    #in the future, can be relative to shoulder-elbow length for diff body types
    
    return abs(left_elbow[1] - middle_knuckle[1]) < 0.10

def swordStraightCheck2(landmarks, hand_landmarks, aspect_ratio):
    left_shoulder = [landmarks[11].x * aspect_ratio, landmarks[11].y, landmarks[11].z]
    left_elbow    = [landmarks[13].x * aspect_ratio, landmarks[13].y, landmarks[13].z]
    left_wrist    = [landmarks[15].x * aspect_ratio, landmarks[15].y, landmarks[15].z]


    elbow_to_shoulder = np.array(left_shoulder) - np.array(left_elbow)
    elbow_to_wrist = np.array(left_wrist) - np.array(left_elbow)

    cosine_angle = np.dot(elbow_to_shoulder, elbow_to_wrist) / (np.linalg.norm(elbow_to_shoulder) * np.linalg.norm(elbow_to_wrist))
    angle = np.arccos(cosine_angle) * 180 / np.pi

    return 60 < angle < 120


#drill detection logic
def main(pose_result, hand_result, aspect_ratio, state, legRaised90):
    """Advances the drill state machine for one frame.

    Returns (state, legRaised90, feedback) so callers hold no hidden state.
    """
    feedback = []

    if not pose_result or not pose_result.pose_landmarks:
        return state, legRaised90, feedback

    landmarks = pose_result.pose_landmarks[0]

    if state == "start":
        isCorrect, feedback = senangDiriCheck(landmarks, aspect_ratio)
        if isCorrect:
            state = "senang_diri"

    #if senang diri, check if transition to leg raised attempt
    elif state == "senang_diri":
        isCorrect, feedback = senangDiriCheck(landmarks, aspect_ratio)
        if not isCorrect:
            state = "start"
        elif legRaisedAttempt(landmarks, aspect_ratio):
            state = "leg_raised_attempt"

    elif state == "leg_raised_attempt":
        isCorrect = legRaisedAttempt(landmarks, aspect_ratio)
        if legRaised90Check(landmarks, aspect_ratio):
            legRaised90 = True
        if not isCorrect:
            if legRaised90:
                state = "sedia"
                legRaised90 = False
            else:
                state = "senang_diri"

    elif state == "sedia":
        isCorrect, feedback = sediaCheck(landmarks, aspect_ratio)

    return state, legRaised90, feedback
