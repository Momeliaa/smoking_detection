import subprocess
import numpy as np
import cv2
import torch
import time

RTMP_URL = "rtmp://43.200.193.228/live/stream"

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.classes = [0]

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

def read_frame(proc):
    y_size = 640 * 480
    uv_size = y_size // 2
    frame_bytes = proc.stdout.read(y_size + uv_size)
    if not frame_bytes:
        return None
    yuv = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((480 * 3) // 2, 640)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

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
    last_detection = 0

    print("YOLO 감지 시작...")

    try:
        while True:
            frame = read_frame(cam_proc)
            if frame is None:
                print("카메라가 연결되지 않음")
                break

            results = model(frame)
            detections = results.pred[0]

            person_detected = False
            for *box, conf, cls in detections:
                if int(cls) == 0:
                    person_detected = True
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            now = time.time()

            if person_detected:
                last_detection = now
                if not streaming:
                    print("사람 감지됨 - RTMP 스트림 시작")
                    ffmpeg_proc = start_rtmp_stream()
                    streaming = True

            elif streaming and (now - last_detection > 5):
                print("사람 감지되지 않음 - RTMP 스트림 종료")
                ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait()
                ffmpeg_proc = None
                streaming = False

            # ffmpeg ����
            if streaming and ffmpeg_proc:
                try:
                    ffmpeg_proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    print("ffmpeg 오류")
                    streaming = False

            cv2.imshow("Preview", frame)
            if cv2.waitKey(1) == 27:  # ESC
                break

    finally:
        cam_proc.terminate()
        if ffmpeg_proc:
            ffmpeg_proc.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
