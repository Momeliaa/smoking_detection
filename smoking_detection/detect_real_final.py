import cv2
from ultralytics import YOLO
import time
import math
import os
import requests
import json
import subprocess
import numpy as np
from datetime import datetime

# ==== 설정 ====
RTMP_URL = "rtmp://localhost/live/stream"
SAVE_DIR = "/var/www/html/frames"
TOKEN_FILE = "device_token.txt"
EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
RASPBERRY_PI_ALERT_URL = "http://43.203.81.52:9001/alert"

# ==== 주소 1회 요청 ====
def fetch_address_from_server():
    try:
        res = requests.get("http://localhost:8000/location")
        data = res.json()
        address = data.get("location", "Unknown")
        print(f"\U0001F4CD 서버에서 수신된 주소: {address}")
        return address
    except Exception as e:
        print("주소 요청 실패:", e)
        return "Unknown"

ADDRESS = fetch_address_from_server()

# ==== 프레임 저장 및 URL 반환 ====
def save_frame_and_get_url(frame):
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"smoking_{timestamp}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(filepath, frame)
    return f"http://43.200.193.228/frames/{filename}"

# ==== Expo Push 알림 전송 ====
def get_device_token():
    try:
        with open(TOKEN_FILE, "r") as f:
            token = f.read().strip()
            if token.startswith("ExponentPushToken"):
                return token
            else:
                print("올바르지 않은 Expo 토큰 형식입니다:", token)
                return None
    except FileNotFoundError:
        print("device_token.txt 없음")
        return None

def send_expo_push_notification(title, body, image_url=None, status="unknown", address="Unknown"):
    token = get_device_token()
    if not token:
        return

    message = {
        "to": token,
        "sound": "default",
        "title": title,
        "body": body,
        "data": {
            "photo": image_url or "",
            "status": status,
            "address": address
        }
    }

    response = requests.post(
        EXPO_PUSH_URL,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json"
        },
        data=json.dumps(message)
    )

    try:
        result = response.json()
        print("\U0001F4E8 푸시 전송 응답:", result)
    except Exception as e:
        print("응답 파싱 실패:", e)

# ==== 라즈베리파이 알림 전송 ====
def notify_raspberry_pi():
    try:
        res = requests.post(RASPBERRY_PI_ALERT_URL, json={"event": "smoking_detected"})
        if res.status_code == 200:
            print("라즈베리파이에 알림 전송 완료")
        else:
            print("라즈베리파이 응답 오류:", res.status_code)
    except Exception as e:
        print("라즈베리파이 알림 전송 실패:", e)

# ==== YOLO 모델 로딩 ====
model = YOLO("best.pt")

# 박스 겹침 확인
def is_overlap(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    return not (x1_1 > x2_2 or x2_1 < x1_2 or y1_1 > y2_2 or y2_1 < y1_2)

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def get_rtmp_frame_generator(rtmp_url):
    proc = subprocess.Popen([
        'ffmpeg', '-i', rtmp_url,
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo', '-'
    ], stdout=subprocess.PIPE)

    w, h = 640, 480
    first_frame = True
    while True:
        raw = proc.stdout.read(w * h * 3)
        if not raw:
            break
        frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3))
        if first_frame:
            print("RTMP 프레임 수신 시작")
            first_frame = False
        yield frame

# ==== 메인 루프 ====
def main():
    retry_interval = 5
    face_tracks = {}
    next_face_id = 0
    MAX_DISTANCE = 150
    MAX_DISAPPEAR = 100
    notified_ids = set()
    smoking_ids = set()

    while True:
        try:
            for frame in get_rtmp_frame_generator(RTMP_URL):
                results = model(frame)
                face_boxes = []
                cigarette_boxes = []

                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls.item())
                        coords = list(map(int, box.xyxy[0]))
                        if cls == 0:
                            face_boxes.append(coords)
                        elif cls == 1:
                            cigarette_boxes.append(coords)

                new_faces = [(box, get_center(box)) for box in face_boxes]

                for fid in list(face_tracks.keys()):
                    face_tracks[fid]['disappear'] += 1
                    if face_tracks[fid]['disappear'] > MAX_DISAPPEAR:
                        del face_tracks[fid]
                        if fid in smoking_ids:
                            print(f"흡연 중지 감지됨! ID: {fid}")
                            send_expo_push_notification(
                                title="흡연 중지 감지",
                                body="더 이상 흡연이 감지되지 않습니다.",
                                image_url="http://43.200.193.228/frames/no_smoking.jpg",
                                status="clear",
                                address=ADDRESS
                            )
                            smoking_ids.remove(fid)
                            notified_ids.discard(fid)

                for face_box, center in new_faces:
                    matched_id = None
                    min_dist = float('inf')

                    for fid, data in face_tracks.items():
                        dist = math.dist(center, data['center'])
                        if dist < MAX_DISTANCE and dist < min_dist:
                            matched_id = fid
                            min_dist = dist

                    if matched_id is not None:
                        face_tracks[matched_id]['center'] = center
                        face_tracks[matched_id]['disappear'] = 0
                    else:
                        face_tracks[next_face_id] = {
                            'center': center,
                            'disappear': 0,
                            'count': 0
                        }
                        matched_id = next_face_id
                        next_face_id += 1

                    for cig_box in cigarette_boxes:
                        if is_overlap(face_box, cig_box):
                            face_tracks[matched_id]['count'] += 1
                            break

                    if face_tracks[matched_id]['count'] >= 30 and matched_id not in notified_ids:
                        print(f"흡연 감지됨! ID: {matched_id}")
                        image_url = save_frame_and_get_url(frame)
                        send_expo_push_notification(
                            title="흡연 감지됨",
                            body="누군가 흡연 중입니다.",
                            image_url=image_url,
                            status="smoking",
                            address=ADDRESS
                        )
                        notify_raspberry_pi()
                        notified_ids.add(matched_id)
                        smoking_ids.add(matched_id)
        except Exception as e:
            print("프레임 수신 오류:", e)
            time.sleep(retry_interval)

if __name__ == "__main__":
    main()
