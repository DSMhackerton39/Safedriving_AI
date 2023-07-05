import os
import cv2
import time
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
import requests
import json

os.environ['OPENBLAS_NUM_THREADS'] = '1'
drowsy_timer_start = None

flag = False

thresholds = {
    "EAR_THRESH": 0.15,
    "WAIT_TIME": 10
}

def plot_eye_landmarks(frame, left_lm_coordinates, right_lm_coordinates, color):
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)

    frame = cv2.flip(frame, 1)
    return frame

def distance(p1, p2):
    dist = sum([(i-j)**2 for i, j in zip(p1, p2)])**0.5
    return dist

def get_ear(landmarks, refer_idxs, frame_w, frame_h):
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_w, frame_h)
            coords_points.append(coord)

        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None
    return ear, coords_points

def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    Avg_EAR = (left_ear + right_ear) / 2.0
    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)

def read_webcam():
    global drowsy_timer_start
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame 처리 및 화면에 표시
        processed_frame = process_frame(frame)
        cv2.imshow("Webcam", processed_frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 프레임을 처리하는 함수
def process_frame(frame):
    global drowsy_timer_start
    global flag
    # 프레임 크기 가져오기
    frame_h, frame_w, _ = frame.shape
    
    # 프레임 전처리
    frame.flags.writeable = False
    
    # 얼굴 및 랜드마크 인식 수행
    results = face_mesh.process(frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 눈의 인덱스
        left_eye_idxs = [362, 385, 387, 263, 373, 380]
        right_eye_idxs = [33, 160, 158, 133, 153, 144]
        
        # 눈 감김 비율 계산
        ear, coordinates = calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, frame_w, frame_h)
        
        # 눈 감김 비율에 따라 색상 및 텍스트 지정
        if ear < thresholds["EAR_THRESH"]:
            if drowsy_timer_start is None:
                drowsy_timer_start = time.time()
            color = (0, 0, 255)  # 빨간색
            text = "Closed eyes state"
            elapsed_time = round(time.time() - drowsy_timer_start, 1)
            if elapsed_time > 5 and not flag:
                flag = True
                requests.post('http://127.0.0.1:8000/', data=json.dumps({"flag": False}))
        else:
            flag = False
            drowsy_timer_start = None
            color = (0, 255, 0)  # 초록색
            text = "Waking state"
            elapsed_time = 0
        
        # 눈의 랜드마크를 그림
        frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], color)
        
        # 화면에 텍스트 표시
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Eyes closed time: {elapsed_time}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

# Mediapipe 얼굴 랜드마크 모델 생성
def get_mediapipe_app(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return face_mesh

if __name__ == '__main__':
    face_mesh = get_mediapipe_app()
    read_webcam()