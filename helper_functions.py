from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import cv2 as cv
import numpy as np


def draw_feedback(frame, lines, color=(255, 0, 0)):
    height, width = frame.shape[:2]
    font_scale = width / 1000
    for i, line in enumerate(lines):
        (text_width, text_height), _ = cv.getTextSize(line, cv.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        x = (width - text_width) // 2
        y = int(height * 0.1) + i * int(text_height * 1.5)
        cv.putText(frame, line, (x, y), cv.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)


def drawSkeleton(image1,result,latest_hand_result):
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