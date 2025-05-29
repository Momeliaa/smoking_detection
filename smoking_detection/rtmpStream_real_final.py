import subprocess
import numpy as np
import cv2
import torch
import time
import requests

# RTMP 및 주소 설정
RTMP_URL = "rtmp://43.200.193.228/live/stream"
LOCATION_API_URL = "http://43.200.193.228:8000/location"
LOCATION_NAME = "한성대학교"

# YOLOv5 모델 로딩 (사람만 감지)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.classes = [0]

# 주소 한 번만 전송
def send_location_data():
    try:
        response = requests.post(LOCATION_API_URL, json={"location": LOCATION_NAME})
        if response.status_code == 200:
            print("📍 주소 전송 완료:", LOCATION_NAME)
        else:
            print("❌ 주소 전송 실패 (응답 코드):", response.status_code)
    except Exception as e:
        print("❌ 주소 전송 예외:", e)

# libcamera-vid subprocess 시작
def start_camera():
    return subprocess.Popen([
        'libcamera-vid',
        '--width', '640',
        '--height', '480',
        '--framerate', '15',
        '--codec', 'yuv420',
        '--nopreview',
        '--timeout', '0',
        '--output', '-'
    ], stdout=subprocess.PIPE, bufsize=10**8)

# YUV 프레임을 BGR로 변환
def read_frame(proc):
    y_size = 640 * 480
    uv_size = y_size // 2
    frame_bytes = proc.stdout.read(y_size + uv_size)
    if not frame_bytes:
        return None
    yuv = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((480 * 3) // 2, 640)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

# ffmpeg RTMP 스트림 subprocess 시작
def start_rtmp_stream():
    return subprocess.Popen([
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', '640x480',
        '-r', '15',
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-f', 'flv',
        RTMP_URL
    ], stdin=subprocess.PIPE)

def main():
    cam_proc = start_camera()
    ffmpeg_proc = None
    streaming = False
    address_sent = False

    print("✅ 카메라 감지 시작...")

    try:
        while True:
            frame = read_frame(cam_proc)
            if frame is None:
                print("⚠️ 프레임 읽기 실패")
                break

            results = model(frame)
            detections = results.pred[0]
            preview_frame = frame.copy()
            person_detected = False

            for *box, conf, cls in detections:
                if int(cls) == 0:
                    person_detected = True
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(preview_frame, f'Person {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if not streaming:
                print("▶️ RTMP 스트림 시작")
                ffmpeg_proc = start_rtmp_stream()
                streaming = True

            if streaming and ffmpeg_proc:
                try:
                    if person_detected:
                        ffmpeg_proc.stdin.write(frame.tobytes())
                        if not address_sent:
                            send_location_data()
                            address_sent = True
                    else:
                        black_frame = np.zeros_like(frame)
                        ffmpeg_proc.stdin.write(black_frame.tobytes())
                except BrokenPipeError:
                    print("❌ ffmpeg 파이프 오류")
                    streaming = False

            # 바운딩 박스가 있는 프리뷰 출력
            cv2.imshow("Preview", preview_frame)
            if cv2.waitKey(1) == 27:
                break

    finally:
        cam_proc.terminate()
        if ffmpeg_proc:
            ffmpeg_proc.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
