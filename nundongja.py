import cv2
import numpy as np
from ultralytics import YOLO
import pygetwindow as gw
from PIL import ImageGrab

# YOLO 모델 로드
model = YOLO("yolo11n.pt")  # YOLOv8n 모델

# 감지할 클래스 목록 (주어진 태그들만 필터링)
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe"
]

# 부분 문자열을 사용하여 창 제목에 "Notepad"가 포함된 창 찾기
window_title_tag = "eyecat"  # 여기에 포함된 태그(부분 문자열)를 입력
# 프로그램 창 찾기
windows = gw.getWindowsWithTitle(window_title_tag)

if not windows:
    windows = gw.getWindowsWithTitle("Chrome")
    print(f"Window with title containing '{window_title_tag}' not found.")

# 첫 번째 창을 선택 (여러 개가 있을 경우)
window = windows[0]
window.activate()  # 창을 활성화

while True:
    # 프로그램 창의 위치와 크기 얻기
    left, top, right, bottom = window.left, window.top, window.right, window.bottom
    
    # 선택한 창의 화면 캡처
    screen = np.array(ImageGrab.grab(bbox=(left, top, right, bottom)))
    frame = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)  # 색상 변환

    # YOLO 객체 감지
    results = model(frame)

    # 바운딩 박스 그리기
    for result in results[0].boxes:  # results[0]에 첫 번째 결과가 저장됨
        x1, y1, x2, y2 = result.xyxy[0].tolist()  # 좌표 추출
        conf = result.conf[0].item()  # 신뢰도
        cls = result.cls[0].item()  # 클래스
        label = model.names[int(cls)]  # 클래스 이름 가져오기

        # 감지된 클래스가 필터링한 클래스 목록에 포함되었는지 확인
        if label in class_names:
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 화면에 결과 표시
    cv2.imshow("YOLO Object Detection", frame)

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cv2.destroyAllWindows()
