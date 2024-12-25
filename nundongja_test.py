import cv2
import numpy as np
from ultralytics import YOLO
import pygetwindow as gw
from PIL import ImageGrab
import win32gui
import win32api
import logging
import time

# 로그 파일 설정
logging.basicConfig(filename='error.log', level=logging.ERROR)

# YOLO 모델 로드
model = YOLO("yolo11n.pt")  # YOLOv8n 모델

# 감지할 클래스 목록 (주어진 태그들만 필터링)
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe"
]

# 부분 문자열을 사용하여 창 제목에 "eyecat"가 포함된 창 찾기
window_title_tag = "eyecat"  # 여기에 포함된 태그(부분 문자열)를 입력
# 프로그램 창 찾기
windows = gw.getWindowsWithTitle(window_title_tag)

if not windows:
    windows = gw.getWindowsWithTitle("Chrome")
    print(f"Window with title containing '{window_title_tag}' not found.")

# 첫 번째 창을 선택 (여러 개가 있을 경우)
window = windows[0]
window.activate()  # 창을 활성화

# 바운딩박스를 그리기 위한 클래스 정의
class Draw:
    def __init__(self):
        hwnd = win32gui.GetDesktopWindow()
        self.hdc = win32gui.GetDC(hwnd)

    def rect(self, x, y, w, h, color=False, thickness=3):
        color = win32api.RGB(0, 255, 0) if not color else win32api.RGB(color[0], color[1], color[2])
        for t in range(thickness):
            for i in range(x - t, x + w + t):
                win32gui.SetPixel(self.hdc, i, y - t, color)
                win32gui.SetPixel(self.hdc, i, y + h + t, color)
            for j in range(y - t, y + h + t):
                win32gui.SetPixel(self.hdc, x - t, j, color)
                win32gui.SetPixel(self.hdc, x + w + t, j, color)

draw = Draw()

try:
    while True:
        try:
            # 프로그램 창의 위치와 크기 얻기
            left, top, right, bottom = window.left, window.top, window.right, window.bottom
            
            # 선택한 창의 화면 캡처
            screen = np.array(ImageGrab.grab(bbox=(left, top, right, bottom)))
            frame = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)  # 색상 변환

            # YOLO 객체 감지
            results = model(frame)

            # 이전에 그린 박스를 지우기 위해 화면을 다시 캡처
            screen = np.array(ImageGrab.grab(bbox=(left, top, right, bottom)))
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

            # 바운딩 박스 그리기
            for result in results[0].boxes:  # results[0]에 첫 번째 결과가 저장됨
                x1, y1, x2, y2 = result.xyxy[0].tolist()  # 좌표 추출
                conf = result.conf[0].item()  # 신뢰도
                cls = result.cls[0].item()  # 클래스
                label = model.names[int(cls)]  # 클래스 이름 가져오기

                # 감지된 클래스가 필터링한 클래스 목록에 포함되었는지 확인
                if label in class_names:
                    # 바운딩 박스 그리기
                    draw.rect(left + int(x1), top + int(y1), int(x2) - int(x1), int(y2) - int(y1), thickness=3)
                    win32gui.TextOut(draw.hdc, left + int(x1), top + int(y1) - 20, f"{label} {conf:.2f}", len(f"{label} {conf:.2f}"))

        except Exception as e:
            logging.error("Exception occurred in loop", exc_info=True)
            print("An error occurred in the loop. Continuing in 0.5 seconds...")
            #time.sleep(0.5)  # 0.5초 대기

        # 'q'를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    logging.error("Exception occurred", exc_info=True)
    input("An error occurred. Press Enter to exit...")  # 예외 발생 시 사용자 입력 대기

finally:
    # 리소스 해제
    win32gui.ReleaseDC(win32gui.GetDesktopWindow(), draw.hdc)
    cv2.destroyAllWindows()
