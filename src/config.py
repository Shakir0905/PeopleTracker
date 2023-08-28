from pathlib import Path

# Основная директория проекта
BASE_DIR = Path(__file__).parent.parent  # предполагая, что config.py находится в папке src

# Определение путей к основным ресурсам
DATA_DIR = BASE_DIR / "data"
YOLO_DIR = DATA_DIR / "yolo"
VIDEOS_DIR = DATA_DIR / "videos"
OUTPUT_VIDEO_PATH = BASE_DIR / "output2.avi"
TRACKED_VIDEOS =  BASE_DIR / "results" / "tracked_videos" 
HOMOGRAPHY_DIR = DATA_DIR / "homography"