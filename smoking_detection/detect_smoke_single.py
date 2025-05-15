import cv2
from ultralytics import YOLO
import time
import math
import sqlite3
from datetime import datetime

# 학습된 모델 불러오기
model = YOLO("runs/detect/train/weights/best.pt")

# SQLite 데이터베이스 초기화
def init_db():
    conn = sqlite3.connect('smoking_detection.db')  # 데이터베이스 연결
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS smoking_detection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_time TEXT,
            smoking_count INTEGER
        )
    ''')
    conn.commit()
    return conn, cursor

# 데이터를 삽입하는 함수
def insert_detection(conn, cursor, smoking_count):
    detected_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO smoking_detection (detected_time, smoking_count)
        VALUES (?, ?)
    ''', (detected_time, smoking_count))
    conn.commit()

# 가장 최근 smoking_count 값을 불러오는 함수
def get_last_smoking_count(cursor):
    cursor.execute('SELECT MAX(smoking_count) FROM smoking_detection')
    result = cursor.fetchone()[0]
    return result if result else 0  # 없으면 0으로 시작


def is_overlap(box1, box2):
    # 각 box는 (x1, y1, x2, y2) 형식
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 두 박스가 겹치지 않는 조건
    if x1_1 > x2_2 or x2_1 < x1_2 or y1_1 > y2_2 or y2_1 < y1_2:
        return False
    return True


def process_video():
    video_path = './tracked_clips/smoking_outside2.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 출력 설정
    output_path = 'C:/Users/jyoung/Desktop/new_result/smoking_outside2_new.mp4'
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 90도 rotate했을 경우
    # out = cv2.VideoWriter(output_path, fourcc, fps, (height, width))

    overlap_count = 0  # 겹침 횟수 카운트
    smoking_detected = False  # 흡연 적발 여부 확인 변수
    
    # SQLite 데이터베이스 초기화
    conn, cursor = init_db()
    smoking_cnt = get_last_smoking_count(cursor)  # DB에서 가장 최근 count 불러오기

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 상하 반전
        # frame = cv2.flip(frame, 0)

        # 오른쪽 90도 회전
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # 프레임에서 객체 감지
        results = model(frame)

        face_box = None
        cigarette_box = None

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls.item())
                coords = list(map(int, box.xyxy[0]))  # [x1, y1, x2, y2]

                if cls == 0:
                    face_box = coords
                elif cls == 1:
                    cigarette_box = coords

                color = (0, 255, 0) if cls == 0 else (255, 0, 0)
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)

        if face_box and cigarette_box:
            if is_overlap(face_box, cigarette_box):
                overlap_count += 1

            if overlap_count >= 30 and not smoking_detected:
                smoking_cnt += 1
                insert_detection(conn, cursor, smoking_cnt)
                smoking_detected = True
                cv2.putText(frame, "Smoking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Overlap Count: {overlap_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        # 프레임 저장
        out.write(frame)
        
        # 화면에 표시 (옵션)
        cv2.imshow('Filtered Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 데이터베이스 연결 종료
    conn.close()

if __name__ == "__main__":
    process_video()