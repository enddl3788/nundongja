import subprocess
import sys

# 설치할 패키지 리스트
packages = ["opencv-python", "numpy", "mss", "ultralytics"]

for package in packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"Failed to install {package}: {e}")
