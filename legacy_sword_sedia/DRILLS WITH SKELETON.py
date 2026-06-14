import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import cv2 as cv
import time
import numpy as np

#boring stuff
cap = cv.VideoCapture(0)
model_path = r"C:\Users\samue\Downloads\pose_landmarker_full.task"
latest_result = None


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarkerResult = vision.PoseLandmarkerResult
RunningMode = vision.RunningMode

feedback = []
isFormCorrect = False
#helper functions
def calcAngle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a-b
    bc = c-b

    dot_product = np.dot(ba,bc)
    magnitude = np.linalg.norm(ba) * np.linalg.norm(bc) 
    angle = np.degrees(np.arccos(dot_product/magnitude))
    return angle
def drawSkeleton(image1,result):
    if not result:
        return 
    if not result.pose_landmarks:
        return
    new_image = np.copy(image1)
    landmarks = result.pose_landmarks[0]
    new_landmarks = [landmarks[3],landmarks[11],landmarks[12],landmarks[13],landmarks[15],landmarks[19] ]

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    my_connections = [
    vision.PoseLandmarksConnections.Connection(start=1, end=2), 
    vision.PoseLandmarksConnections.Connection(start=1, end=3),  
    vision.PoseLandmarksConnections.Connection(start=3, end=4),
    vision.PoseLandmarksConnections.Connection(start=4, end=5),  
    
]
    drawing_utils.draw_landmarks(
            image=new_image,
            landmark_list=new_landmarks,
            connections=my_connections,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style)
    return new_image



#drill detection logic
def callbackFunction(result,image,timestamp):
    global feedback, isFormCorrect,latest_result,sword_detected
    if not result.pose_landmarks:
        return
    latest_result = result
    landmarks = result.pose_landmarks[0]
    
    #correct for different dimensions
    height,width, = image.height,image.width
    aspect_ratio = width/height

    #right and left flipped to account for mirror image from webcam
    right_shoulder = [landmarks[11].x*aspect_ratio, landmarks[11].y]
    left_shoulder = [landmarks[12].x*aspect_ratio, landmarks[12].y]
    right_elbow = [landmarks[13].x*aspect_ratio, landmarks[13].y]
    right_wrist = [landmarks[15].x*aspect_ratio, landmarks[15].y]
    right_index = [landmarks[19].x*aspect_ratio, landmarks[19].y]
    right_eye = [landmarks[3].x*aspect_ratio, landmarks[3].y]

    #arm at 90 degrees
    upper_arm_angle = calcAngle(left_shoulder, right_shoulder, right_elbow)
    isUpperArmPerpendicular = 170 <= upper_arm_angle <= 180
    
    #forearm and hand straight
    forearm_angle = calcAngle(right_elbow,right_wrist,right_index)
    isForearmStraight = 160 <= forearm_angle <= 180

    #fingertip next to eyebrow(using eye for now)
    dist = np.linalg.norm(np.array(right_index) - np.array(right_eye))
    isFingerAtEyebrow = dist < 0.10

    errors = []
    if not isUpperArmPerpendicular:
        errors.append("Arm not perpendicular")
    if not isForearmStraight:
        errors.append("Forearm not straight")
    if not isFingerAtEyebrow:
        errors.append("Finger not at eyebrow")

    feedback = errors
    isFormCorrect = isUpperArmPerpendicular and isForearmStraight and isFingerAtEyebrow



#main
options = PoseLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=model_path),
    running_mode = RunningMode.LIVE_STREAM,
    result_callback = callbackFunction,
)

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        working,frame = cap.read()
        if not working:
            continue
        frame = cv.flip(frame, 1)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(time.time()*1000)
        landmarker.detect_async(mp_image, timestamp)
        result = latest_result


        #display
        height, width, _ = frame.shape
        color = (0, 255, 0) if isFormCorrect else (255, 0, 0)
        cv.rectangle(frame, (0, 0), (width, height), color, 20)

        font_scale = width / 1000
        for i, line in enumerate(feedback):
            (text_width, text_height), _ = cv.getTextSize(line, cv.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            x = (width - text_width) // 2
            y = int(height * 0.1) + i * int(text_height * 1.5)
            cv.putText(frame, line, (x, y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)
        
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        annotated_frame = drawSkeleton(frame,result) 
        if annotated_frame is None:
            annotated_frame = frame

        cv.imshow("Drill Checker", annotated_frame) 
        if cv.waitKey(1) == ord('q'):
            break
    
cap.release()
cv.destroyAllWindows()