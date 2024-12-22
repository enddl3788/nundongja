import sys
import cv2
import numpy as np
from ultralytics import YOLO
import mss
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

# YOLO 모델 로드
model = YOLO("yolov8n.pt")  # YOLOv8n 모델

# 화면 캡처 설정
sct = mss.mss()
monitor = sct.monitors[1]  # 첫 번째 모니터

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Object Detection")
        self.setGeometry(0, 0, 800, 600)

        # QLabel을 사용하여 이미지 표시
        self.label = QLabel(self)
        self.label.resize(800, 600)

        # 타이머를 사용하여 화면을 주기적으로 업데이트
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)  # 30ms마다 화면 업데이트

    def update_frame(self):
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

        # OpenCV 이미지를 QImage로 변환
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)

        # QLabel에 이미지 표시
        self.label.setPixmap(QPixmap.fromImage(q_img))

# PyQt 애플리케이션 실행
app = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(app.exec_())
