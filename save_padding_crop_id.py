import torch
import cv2
import numpy as np
import os
import warnings
import time
import math  # hypot 함수 사용을 위해 추가

warnings.filterwarnings('ignore', category=FutureWarning)

# ===========================
# 1. 설정 (Configuration)
# ===========================
OPTIMIZATION_CONFIG = {
    'frame_skip_interval': 1,  # 몇 프레임마다 1번 처리할지 (1이면 모두 처리)
    'inference_size': 320  # 모델 추론 시 크기 (라즈베리파이 고려하여 작게 설정)
}

MODEL_CONFIG = {
    'model_type': 'yolov5s',  # 사용할 YOLOv5 모델
    'conf_thres': 0.35,  # 신뢰도 임계값 (이 값 이상만 객체로 간주)
    'iou_thres': 0.45,  # IoU 임계값 (NMS에서 사용, 겹치는 박스 제거 기준)
    'augment': False  # Test Time Augmentation 사용 여부
}

VIS_CONFIG = {
    'line_thickness': 1,  # 바운딩 박스 선 두께
    'font_scale': 0.5,    # 텍스트 폰트 크기
    'font_thickness': 1,  # 텍스트 폰트 두께
    'display_labels': True, # 클래스 이름 표시 여부
    'display_conf': False,  # 신뢰도 표시 여부 (ID를 표시하므로 신뢰도는 생략 가능)
    'box_color': (0, 255, 0),    # 바운딩 박스 색상 (BGR)
    'text_color': (0, 0, 0),     # 텍스트 색상 (BGR)
    'text_bg_color': (0, 255, 0) # 텍스트 배경 색상 (BGR)
}

TRACKING_CONFIG = {
    'max_disappeared_frames': 20,  # 객체가 사라진 것으로 간주하기까지의 최대 프레임 수
    'max_distance': 75  # 객체를 동일 객체로 매칭하기 위한 최대 픽셀 거리 (inference_size에 따라 조절)
}

TARGET_CLASS_NAME = 'person' # 추적할 특정 객체 클래스 이름
VIDEO_PATH = 'C:/Users/jyoung/Desktop/test/smoking_outside2.mp4'  # 예시 비디오 경로, 0으로 하면 웹캠 사용
# VIDEO_PATH = 0 # 웹캠 사용시

# ===========================
# 2. 헬퍼 함수 (Helpers)
# ===========================
def should_skip_frame(frame_count, interval):
    """프레임 건너뛰기 여부를 결정합니다."""
    if interval <= 1:
        return False
    return frame_count % interval != 0


def optimize_model_quantization(model, device):
    """CPU 환경에서 모델 동적 양자화를 시도합니다."""
    if device.type == 'cpu':
        try:
            print("CPU 환경: 동적 양자화 적용 중...")
            # Linear 레이어에 대해 동적 양자화 적용
            qmodel = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            qmodel.eval() # 양자화 후 평가 모드로 설정
            print("동적 양자화 완료.")
            return qmodel
        except Exception as e:
            print(f"양자화 실패: {e} → 원본 모델 사용")
            return model
    else:
        print("GPU 환경: 양자화 건너뜀")
        return model


def preprocess_frame_for_inference(frame, target_size):
    """모델 추론을 위해 프레임 크기를 조절하고 스케일링 비율을 반환합니다."""
    h, w = frame.shape[:2]
    # 가로, 세로 중 긴 쪽을 기준으로 target_size에 맞게 비율 계산
    scale = target_size / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    # 원본 프레임 좌표로 복원하기 위한 스케일링 비율 반환
    return resized, w / nw, h / nh


def get_centroid(box):
    """바운딩 박스의 중심점(centroid)을 계산합니다."""
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def draw_tracked_objects(frame, tracked_items, class_names, vis_config):
    """
    추적된 객체들 (ID 포함)을 프레임에 그립니다.
    tracked_items: list of (object_id, box, conf, class_id, scaled_box)
                   scaled_box는 원본 프레임 좌표의 박스
    """
    for item in tracked_items:
        object_id, _original_box, conf, class_id, scaled_box = item  # _original_box는 추론 크기 기준 박스

        x1, y1, x2, y2 = map(int, scaled_box) # 원본 프레임 스케일의 박스 좌표

        # 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), vis_config['box_color'], vis_config['line_thickness'])

        # 라벨 텍스트 준비 (ID 포함)
        label_parts = []
        label_parts.append(f"ID:{object_id}") # 객체 ID 추가

        if vis_config['display_labels']:
            # 클래스 ID를 이름으로 변환
            name = class_names[int(class_id)] if int(class_id) < len(class_names) else f"ClassID{int(class_id)}"
            label_parts.append(name)
        if vis_config['display_conf']:
            label_parts.append(f"{conf:.2f}") # 신뢰도 추가

        txt = ' '.join(label_parts) # 모든 정보 합치기

        if txt:
            # 텍스트 크기 계산
            (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                             vis_config['font_scale'], vis_config['font_thickness'])
            # 텍스트 배경 박스 좌표 계산
            y0 = max(y1 - th - base, 0) # 프레임 상단을 벗어나지 않도록
            cv2.rectangle(frame, (x1, y0), (x1 + tw, y1), vis_config['text_bg_color'], -1) # 배경 채우기
            # 텍스트 쓰기
            cv2.putText(frame, txt, (x1, y1 - base),
                        cv2.FONT_HERSHEY_SIMPLEX, vis_config['font_scale'],
                        vis_config['text_color'], vis_config['font_thickness'], cv2.LINE_AA)





# ===========================
# 3. 모델 로드 및 최적화
# ===========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}")

# YOLOv5 모델 로드 (PyTorch Hub 사용)
model = torch.hub.load('ultralytics/yolov5', MODEL_CONFIG['model_type'], pretrained=True)
model = model.to(device).eval() # 모델을 해당 장치로 옮기고 평가 모드로 설정
# 모델 추론 설정 적용
model.conf = MODEL_CONFIG['conf_thres']
model.iou = MODEL_CONFIG['iou_thres']
model.augment = MODEL_CONFIG['augment']

if device.type == 'cpu':  # CPU에서만 양자화 시도
    model = optimize_model_quantization(model, device)

# 모델의 클래스 이름들 가져오기
class_names = model.names if isinstance(model.names, list) else list(model.names.values())
try:
    # 추적 대상 클래스의 ID 찾기
    target_class_id = class_names.index(TARGET_CLASS_NAME)
    print(f"Target '{TARGET_CLASS_NAME}' ID: {target_class_id}")
except ValueError:
    print(f"Error: Target class '{TARGET_CLASS_NAME}' not found in model classes: {class_names}")
    target_class_id = None # 대상 클래스가 없으면 None으로 설정
    # exit() # 필요시 프로그램 종료

# ===========================
# 4. 동영상 처리 루프
# ===========================
cap = cv2.VideoCapture(VIDEO_PATH) # 비디오 파일 또는 웹캠 열기
if not cap.isOpened():
    raise SystemExit(f"Cannot open video {VIDEO_PATH}")

# 원본 비디오 정보
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
orig_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {orig_w}×{orig_h} @ {orig_fps:.1f}FPS")

frame_cnt = 0 # 현재 프레임 카운터

# --- 객체 추적 관련 변수 초기화 ---
# tracked_objects: 추적 중인 객체 정보를 저장하는 딕셔너리
#   key: object_id
#   value: [cx, cy, last_seen_frame, disappeared_frames, box_proc, conf, class_id]
#     cx, cy: 추론 크기 기준 중심점 x, y 좌표
#     last_seen_frame: 마지막으로 해당 객체가 관찰된 프레임 번호
#     disappeared_frames: 연속적으로 관찰되지 않은 프레임 수
#     box_proc: 추론 크기 기준 바운딩 박스 (x1, y1, x2, y2)
#     conf: 신뢰도
#     class_id: 클래스 ID
tracked_objects = {}
next_object_id = 0 # 다음에 할당할 객체 ID
last_drawn_tracked_items = []  # 프레임 건너뛰기 시 사용할 마지막 추적 객체 정보

while True:
    ret, frame = cap.read() # 프레임 읽기
    if not ret: # 비디오의 끝이거나 오류 발생 시
        break

    current_tracked_items_to_draw = []  # 현재 프레임에 그릴 객체 정보 리스트

    if should_skip_frame(frame_cnt, OPTIMIZATION_CONFIG['frame_skip_interval']):
        # 건너뛰는 프레임: 이전 추적 결과 사용
        current_tracked_items_to_draw = last_drawn_tracked_items
    else:
        # 처리할 프레임
        t0 = time.time() # 처리 시작 시간 기록
        # 프레임 전처리 (크기 조절)
        proc_frame, scale_w, scale_h = preprocess_frame_for_inference(frame, OPTIMIZATION_CONFIG['inference_size'])
        rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB) # BGR -> RGB 변환

        # 모델 추론
        with torch.no_grad(): # 그래디언트 계산 비활성화
            results = model(rgb_frame, size=OPTIMIZATION_CONFIG['inference_size'])

        detections_raw = results.xyxy[0].cpu().numpy() # 결과 텐서를 CPU numpy 배열로 변환

        # 현재 프레임에서 'person' 클래스로 탐지된 객체들 정보 저장
        current_person_detections = []  # (centroid, box_proc, conf, class_id) for 'person'
        for *box, conf, cid_float in detections_raw:
            cid = int(cid_float)
            if target_class_id is not None and cid == target_class_id: # 설정된 타겟 클래스만 필터링
                # box는 추론 크기(proc_frame) 기준 좌표
                x1, y1, x2, y2 = map(int, box)
                centroid = get_centroid((x1, y1, x2, y2)) # 중심점 계산
                current_person_detections.append((centroid, (x1, y1, x2, y2), conf, cid))

        # --- ID 할당 및 추적 로직 (Centroid Tracking) ---
        # 이 부분에 대한 자세한 설명은 코드 아래에 있습니다.

        # 현재 프레임에서 탐지된 객체들의 ID (아직 할당되지 않음, 매칭/등록 후 채워짐)
        object_ids_this_frame = []

        # 임시로 사용할 현재 프레임 객체 리스트 (매칭되지 않은 객체들)
        unmatched_detections = list(current_person_detections)

        # 기존 추적 객체들을 순회하며 매칭 시도
        # processed_tracked_ids: 현재 프레임에서 매칭되거나 업데이트된 기존 추적 객체의 ID 집합
        processed_tracked_ids = set()

        if tracked_objects:  # 추적 중인 객체가 있을 때만 매칭 시도
            # 기존 객체와 현재 탐지된 객체 간 거리 계산 및 매칭
            # (더 복잡한 매칭은 Hungarian algorithm 등을 사용할 수 있음)
            temp_tracked_objects = list(tracked_objects.items())  # 반복 중 딕셔너리 변경을 피하기 위해 복사본 사용

            for obj_id, data in temp_tracked_objects:
                prev_cx, prev_cy, _last_seen, _disappeared, _box, _conf, _cid = data

                best_match_dist = TRACKING_CONFIG['max_distance'] # 최대 허용 거리
                best_match_idx = -1 # 가장 잘 맞는 unmatched_detections의 인덱스

                # 현재 프레임에서 탐지된 객체들(unmatched_detections)과 거리 비교
                for i, (current_centroid, current_box, current_conf, current_cid) in enumerate(unmatched_detections):
                    current_cx, current_cy = current_centroid
                    dist = math.hypot(current_cx - prev_cx, current_cy - prev_cy) # 유클리드 거리

                    if dist < best_match_dist: # 더 가까운 객체를 찾으면
                        best_match_dist = dist
                        best_match_idx = i

                if best_match_idx != -1:  # 매칭 성공 (허용 거리 내에 객체 발견)
                    # 매칭된 객체를 unmatched_detections에서 제거하고 정보 가져오기
                    (matched_centroid, matched_box, matched_conf, matched_cid) = unmatched_detections.pop(best_match_idx)
                    # tracked_objects 딕셔너리 업데이트: 중심점, 마지막 관찰 프레임, 사라진 프레임 수(0으로 초기화), 박스, 신뢰도, 클래스 ID
                    tracked_objects[obj_id] = [matched_centroid[0], matched_centroid[1], frame_cnt, 0, matched_box,
                                               matched_conf, matched_cid]
                    object_ids_this_frame.append(obj_id) # 현재 프레임에 나타난 객체 ID로 추가
                    processed_tracked_ids.add(obj_id) # 이 ID는 현재 프레임에서 처리됨

        # 매칭되지 않은 나머지 탐지된 객체들은 새로운 ID로 등록
        for (centroid, box, conf, cid) in unmatched_detections:
            tracked_objects[next_object_id] = [centroid[0], centroid[1], frame_cnt, 0, box, conf, cid]
            object_ids_this_frame.append(next_object_id)
            processed_tracked_ids.add(next_object_id) # 새로 등록된 ID도 현재 프레임에서 처리됨
            next_object_id += 1
            if next_object_id > 10000: next_object_id = 0  # ID 재활용 (간단한 방식, 충돌 가능성 있음)

        # 현재 프레임에서 보이지 않은 기존 객체들의 disappeared_count 증가 및 제거
        ids_to_deregister = [] # 제거할 객체 ID 리스트
        # list()로 복사본을 만들어 순회 (딕셔너리 변경 중 에러 방지)
        for obj_id, data in list(tracked_objects.items()):
            if obj_id not in processed_tracked_ids:  # 현재 프레임에서 매칭/갱신되지 않은 (즉, 보이지 않은) 객체
                tracked_objects[obj_id][3] += 1  # disappeared_frames 증가
                if tracked_objects[obj_id][3] > TRACKING_CONFIG['max_disappeared_frames']:
                    ids_to_deregister.append(obj_id) # 최대 허용치를 넘으면 제거 목록에 추가
            # else: # 매칭된 경우, last_seen_frame 업데이트 (이미 위에서 갱신됨)
            #    tracked_objects[obj_id][2] = frame_cnt

        for obj_id in ids_to_deregister: # 제거 목록에 있는 객체들 삭제
            # print(f"Deregistering object ID {obj_id}")
            del tracked_objects[obj_id]

        # --- 현재 프레임에 그릴 객체 정보 리스트 생성 ---
        for obj_id, data in tracked_objects.items():
            # 현재 프레임에 보이는 객체 (disappeared_frames == 0)만 그리기 대상
            if data[3] == 0:
                _cx, _cy, _lsf, _df, box_proc, conf, cid = data
                # 원본 프레임 좌표로 박스 스케일링
                scaled_box = (box_proc[0] * scale_w, box_proc[1] * scale_h,
                              box_proc[2] * scale_w, box_proc[3] * scale_h)
                current_tracked_items_to_draw.append((obj_id, box_proc, conf, cid, scaled_box))

        last_drawn_tracked_items = current_tracked_items_to_draw  # 다음 건너뛰기 프레임 위해 저장

        # 처리 시간 및 FPS 정보 표시
        dur = time.time() - t0
        fps_now = 1.0 / dur if dur > 0 else 0
        cv2.putText(frame, f"Frame:{frame_cnt} ({proc_frame.shape[1]}x{proc_frame.shape[0]})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS:{fps_now:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracked:{len(tracked_objects)}", # 현재 추적 중인 총 객체 수
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 생성된 current_tracked_items_to_draw 리스트를 사용하여 화면에 그리기
    draw_tracked_objects(frame, current_tracked_items_to_draw, class_names, VIS_CONFIG)

    # 객체별 bounding box 영역을 crop, resize, 저장
    CROP_SIZE = (160, 320)  # 원하는 크기로 조절 (예: 160x320)
    save_dir = 'tracked_clips'
    os.makedirs(save_dir, exist_ok=True)

    # 객체별 VideoWriter 저장기 초기화 딕셔너리
    if 'video_writers' not in globals():
        video_writers = {}

    for item in current_tracked_items_to_draw:
        obj_id, _box_proc, _conf, _cid, scaled_box = item
        x1, y1, x2, y2 = map(int, scaled_box)

        # ROI 잘라내기
        roi = frame[max(0, y1):min(y2, frame.shape[0]), max(0, x1):min(x2, frame.shape[1])]
        if roi.size == 0:
            continue  # 잘못된 ROI는 무시

        roi_resized = cv2.resize(roi, CROP_SIZE)

        # VideoWriter 초기화
        if obj_id not in video_writers:
            save_path = os.path.join(save_dir, f"id_{obj_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writers[obj_id] = cv2.VideoWriter(save_path, fourcc, orig_fps, CROP_SIZE)

        # 프레임 저장
        video_writers[obj_id].write(roi_resized)

    cv2.imshow("YOLOv5 with ID Tracking", frame) # 결과 프레임 보여주기
    frame_cnt += 1 # 프레임 카운터 증가
    if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' 키 누르면 종료
        break

cap.release() # 비디오 캡처 객체 해제
cv2.destroyAllWindows() # 모든 OpenCV 창 닫기2