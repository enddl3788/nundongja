import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    """Install all required packages."""
    packages = [
        "torch",
        "opencv-python-headless",
        "ultralytics",
        "mss",
        "pygetwindow",
        "pillow",
        "pyqt5"
    ]

    for package in packages:
        try:
            print(f"Installing {package}...")
            install_package(package)
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            continue
    print("All packages installed successfully.")

if __name__ == "__main__":
    main()
