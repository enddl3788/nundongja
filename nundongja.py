import cv2
import numpy as np
from ultralytics import YOLO
import mss

# YOLO 모델 로드
model = YOLO("yolov10n.pt")  # YOLOv8n 모델

# 화면 캡처 설정
sct = mss.mss()
monitor = sct.monitors[1]  # 첫 번째 모니터

while True:
    # 화면 캡처
    screen = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

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

    # 화면에 바로 표시 (OpenCV 윈도우를 사용하지 않음)
    cv2.imshow("YOLO Object Detection", frame)

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cv2.destroyAllWindows()
sct.close()
