import cv2
import numpy as np
from ultralytics import YOLO
import pygetwindow as gw
from PIL import ImageGrab

# YOLO 모델 로드
model = YOLO("yolo11n.pt")  # YOLOv8n 모델

# 캡처할 창 선택 (예시: 'Untitled - Notepad' 창을 선택)
window_title = "Castro Street Cam 1 Live Stream - Youtube - Chrome"  # 여기에 선택할 프로그램 창의 제목을 입력

# 프로그램 창 찾기
window = gw.getWindowsWithTitle(window_title)

if not window:
    print(f"Window with title '{window_title}' not found.")
    exit()

window = window[0]  # 첫 번째 창을 선택
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

        # 바운딩 박스 그리기
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 화면에 결과 표시
    cv2.imshow("YOLO Object Detection", frame)

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cv2.destroyAllWindows()
