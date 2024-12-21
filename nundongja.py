import cv2
import numpy as np
import mss
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO('yolov8n.pt')  # 사전 학습된 YOLOv8 모델

# 화면 캡처 객체 생성
sct = mss.mss()

# 모니터 크기 확인 (전체 화면 캡처)
monitor = sct.monitors[1]  # 첫 번째 모니터 (모니터 번호는 시스템에 따라 다를 수 있음)

while True:
    # 화면 캡처
    screenshot = sct.grab(monitor)
    
    # 캡처된 이미지를 NumPy 배열로 변환
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)  # OpenCV는 BGR 형식을 사용
    
    # YOLO로 객체 탐지
    results = model(img, conf=0.5)

    # 탐지된 객체에 박스 그리기
    for box in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = box
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 화면 출력
    cv2.imshow("Screen Capture Object Detection", img)

    # 종료 키 설정 (q 키 누르면 종료)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
