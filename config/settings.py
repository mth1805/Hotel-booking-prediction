import os
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "hotel_bookings.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
IMG_DIR = os.path.join(REPORT_DIR, "images")
LOG_FILE = os.path.join(REPORT_DIR, "eda_activity.log")

os.makedirs(IMG_DIR, exist_ok=True)

COLORS = {
    "main_color": "#3498db",
    "bg_color": "#e3f2fd",
    "warn_bg": "#ffe6e6",
    "text_black": "#333333"
}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )