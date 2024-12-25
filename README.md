# 프로젝트: YOLO를 활용한 실시간 CCTV 객체 감지

## 소개
이 프로젝트는 YOLO (You Only Look Once) 모델과 Python을 사용하여 CCTV 화면에서 실시간으로 객체(자동차, 사람 등)를 감지하고, 해당 객체에 대한 바운딩 박스를 화면에 표시합니다. 이 프로그램은 mss를 사용하여 화면을 캡처하고 OpenCV를 통해 바운딩 박스를 그립니다.

---

## 기능
- YOLOv8 모델을 활용한 객체 감지
- mss를 사용하여 실시간으로 모니터 화면 캡처
- OpenCV를 이용한 객체 감지 결과 화면 출력
- PyQt5를 사용하여 GUI 기반의 객체 감지 결과 표시 가능 (선택 사항)
- 간단한 설정으로 다양한 모니터 환경에서 작동 가능

---

## 설치 및 실행 방법

### 1. Python 설치
- Python 3.10.0 버전을 설치합니다.

### 2. 필수 라이브러리 설치
아래 명령어를 사용하여 필요한 라이브러리를 설치합니다:

```bash
pip install -U pip
pip install opencv-python-headless ultralytics mss pyqt5
```

### 3. YOLO 모델 다운로드
YOLOv8 모델을 다운로드하여 사용할 수 있도록 설정합니다:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
```

### 4. 프로그램 실행
`nundongja.py` 파일을 실행하여 프로그램을 시작합니다:

```bash
python nundongja.py
```

---

## 코드 설명

### 주요 라이브러리
- **OpenCV**: 객체 감지 결과를 화면에 표시하기 위한 라이브러리
- **mss**: 모니터 화면을 캡처하기 위한 라이브러리
- **PyQt5**: GUI를 사용하여 객체 감지 결과를 표시하기 위한 선택적 라이브러리
- **YOLO**: YOLOv8 모델을 사용하여 객체를 감지

### 코드 구조

#### 1. YOLO 모델 로드
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
```

#### 2. 화면 캡처
```python
import mss
sct = mss.mss()
monitor = sct.monitors[1]  # 첫 번째 모니터
```

#### 3. 객체 감지 및 바운딩 박스 그리기
```python
for result in results[0].boxes:
    x1, y1, x2, y2 = result.xyxy[0].tolist()
    conf = result.conf[0].item()
    cls = result.cls[0].item()
    label = model.names[int(cls)]

    # 바운딩 박스 그리기
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

#### 4. 화면에 결과 표시
```python
cv2.imshow("YOLO Object Detection", frame)
```

---

## 문제 해결 과정

1. **Python 버전 문제**
   - Python 3.10.0으로 변경하여 호환성 문제 해결.

2. **필수 라이브러리 설치**
   - `pip`를 최신 버전으로 업데이트하고 필요한 라이브러리 설치.

3. **YOLO 객체 감지 오류**
   - `results[0].boxes` 형식으로 변경하여 객체 감지 결과를 올바르게 처리.

4. **무한 화면 문제**
   - PyQt5를 사용하거나 OpenCV 윈도우 창을 적절히 처리하여 해결.

---

## 주요 참고 자료
- [YOLO 공식 문서](https://docs.ultralytics.com/)
- [OpenCV 공식 문서](https://docs.opencv.org/)
- [PyQt5 공식 문서](https://riverbankcomputing.com/software/pyqt/intro)

---

## 개선 및 추가 작업
- 다중 모니터 환경 지원 개선
- 감지 속도 최적화
- PyQt5 GUI 통합으로 사용자 경험 향상

---

## 종료
- `q` 키를 눌러 프로그램 종료.

---

### 감사합니다!
