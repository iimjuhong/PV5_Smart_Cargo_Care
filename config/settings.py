"""
Global Configuration Settings for Smart Cargo Care System
전역 설정 - 모든 임계값과 시스템 파라미터를 관리합니다

환경변수를 통해 설정을 오버라이드할 수 있습니다.
예: SMART_CARGO_CONFIDENCE_THRESHOLD=0.7
"""

import os
from typing import Union


def get_env_float(key: str, default: float) -> float:
    """환경변수에서 float 값을 가져옴"""
    value = os.environ.get(f"SMART_CARGO_{key}")
    if value is not None:
        try:
            return float(value)
        except ValueError:
            pass
    return default


def get_env_int(key: str, default: int) -> int:
    """환경변수에서 int 값을 가져옴"""
    value = os.environ.get(f"SMART_CARGO_{key}")
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass
    return default


def get_env_bool(key: str, default: bool) -> bool:
    """환경변수에서 bool 값을 가져옴"""
    value = os.environ.get(f"SMART_CARGO_{key}")
    if value is not None:
        return value.lower() in ("true", "1", "yes")
    return default


def get_env_str(key: str, default: str) -> str:
    """환경변수에서 str 값을 가져옴"""
    return os.environ.get(f"SMART_CARGO_{key}", default)


# ==================== YOLO Model Settings ====================
# 모델 저장 디렉토리
MODELS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# 기본 YOLO 모델 (일반 객체 검출)
MODEL_PATH: str = get_env_str("MODEL_PATH", os.path.join(MODELS_DIR, "yolov8n.pt"))

# Waste Classification 모델 (cardboard 등 12개 클래스)
WASTE_MODEL_PATH: str = get_env_str("WASTE_MODEL_PATH", os.path.join(MODELS_DIR, "waste-classification.pt"))
WASTE_MODEL_HF_ID: str = "kendrickfff/waste-classification-yolov8-ken"

# OBB 모델 (회전 바운딩 박스 - 쓰러짐 감지용)
OBB_MODEL_PATH: str = get_env_str("OBB_MODEL_PATH", os.path.join(MODELS_DIR, "yolov8n-obb.pt"))

# 검출 설정
CONFIDENCE_THRESHOLD: float = get_env_float("CONFIDENCE_THRESHOLD", 0.5)
IOU_THRESHOLD: float = get_env_float("IOU_THRESHOLD", 0.5)

# 설정값 검증
assert 0.0 <= CONFIDENCE_THRESHOLD <= 1.0, "CONFIDENCE_THRESHOLD must be between 0.0 and 1.0"
assert 0.0 <= IOU_THRESHOLD <= 1.0, "IOU_THRESHOLD must be between 0.0 and 1.0"

# ==================== Anomaly Detection Thresholds ====================
# 중심점 이동 임계값 (픽셀 단위)
MOVEMENT_THRESHOLD: float = get_env_float("MOVEMENT_THRESHOLD", 30.0)

# 종횡비 변화율 임계값 (0.0 ~ 1.0)
ASPECT_RATIO_THRESHOLD: float = get_env_float("ASPECT_RATIO_THRESHOLD", 0.15)

# 추적 임계값
TRACKING_IOU_THRESHOLD: float = get_env_float("TRACKING_IOU_THRESHOLD", 0.5)
MAX_DISAPPEARED_FRAMES: int = get_env_int("MAX_DISAPPEARED_FRAMES", 30)

# 설정값 검증
assert MOVEMENT_THRESHOLD >= 0.0, "MOVEMENT_THRESHOLD must be non-negative"
assert 0.0 <= ASPECT_RATIO_THRESHOLD <= 1.0, "ASPECT_RATIO_THRESHOLD must be between 0.0 and 1.0"
assert 0.0 <= TRACKING_IOU_THRESHOLD <= 1.0, "TRACKING_IOU_THRESHOLD must be between 0.0 and 1.0"
assert MAX_DISAPPEARED_FRAMES > 0, "MAX_DISAPPEARED_FRAMES must be positive"

# ==================== Camera Settings ====================
# 카메라 소스 설정
# 옵션 1: 스마트폰 (DroidCam, IP Webcam 등)
# 옵션 2: 일반 웹캠 (정수로 설정: "0", "1", "2")
_camera_source_env = os.environ.get("SMART_CARGO_CAMERA_SOURCE")
if _camera_source_env is not None:
    # 정수인 경우 (웹캠 인덱스)
    try:
        CAMERA_SOURCE: Union[int, str] = int(_camera_source_env)
    except ValueError:
        CAMERA_SOURCE: Union[int, str] = _camera_source_env
else:
    CAMERA_SOURCE: Union[int, str] = "http://192.168.0.10:8080/video"

# 프레임 해상도
FRAME_WIDTH: int = get_env_int("FRAME_WIDTH", 640)
FRAME_HEIGHT: int = get_env_int("FRAME_HEIGHT", 480)

# FPS 제한 (0 = 제한 없음)
FPS_LIMIT: int = get_env_int("FPS_LIMIT", 30)

# 설정값 검증
assert FRAME_WIDTH > 0, "FRAME_WIDTH must be positive"
assert FRAME_HEIGHT > 0, "FRAME_HEIGHT must be positive"
assert FPS_LIMIT >= 0, "FPS_LIMIT must be non-negative"

# ==================== UART Communication Settings ====================
# UART 포트 설정
# macOS 예시: "/dev/cu.usbserial-1420" 또는 "/dev/tty.usbserial-1420"
# Linux 예시: "/dev/ttyUSB0" 또는 "/dev/ttyACM0"
# Windows 예시: "COM3"
UART_PORT: str = get_env_str("UART_PORT", "/dev/cu.usbserial-1420")

# Baud Rate 설정
UART_BAUDRATE: int = get_env_int("UART_BAUDRATE", 9600)

# 통신 타임아웃 (초)
UART_TIMEOUT: float = get_env_float("UART_TIMEOUT", 1.0)

# UART 활성화 여부 (테스트 시 False로 설정 가능)
UART_ENABLED: bool = get_env_bool("UART_ENABLED", True)

# 설정값 검증
assert UART_BAUDRATE > 0, "UART_BAUDRATE must be positive"
assert UART_TIMEOUT > 0.0, "UART_TIMEOUT must be positive"

# ==================== Logging Settings ====================
LOG_LEVEL: str = get_env_str("LOG_LEVEL", "INFO")
LOG_TO_FILE: bool = get_env_bool("LOG_TO_FILE", True)
LOG_FILE_PATH: str = get_env_str("LOG_FILE_PATH", "logs/smart_cargo.log")
LOG_MAX_BYTES: int = get_env_int("LOG_MAX_BYTES", 10 * 1024 * 1024)  # 10MB
LOG_BACKUP_COUNT: int = get_env_int("LOG_BACKUP_COUNT", 5)

# 설정값 검증
assert LOG_LEVEL in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), \
    "LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL"

# ==================== Visualization Settings ====================
# 디버깅용 실시간 시각화 활성화
ENABLE_VISUALIZATION: bool = get_env_bool("ENABLE_VISUALIZATION", True)

# 시각화 윈도우 이름
WINDOW_NAME: str = get_env_str("WINDOW_NAME", "Smart Cargo Care - Box Detection")

# 박스 색상 (BGR 형식)
COLOR_NORMAL: tuple = (0, 255, 0)  # 초록색 - 정상
COLOR_MOVEMENT: tuple = (0, 165, 255)  # 주황색 - 이동 감지
COLOR_COLLISION: tuple = (0, 0, 255)  # 빨간색 - 충돌 감지

# ==================== System Settings ====================
# 모델 자동 다운로드 활성화
AUTO_DOWNLOAD_MODEL: bool = get_env_bool("AUTO_DOWNLOAD_MODEL", True)

# 성능 모니터링
SHOW_FPS: bool = get_env_bool("SHOW_FPS", True)
SHOW_DETECTION_COUNT: bool = get_env_bool("SHOW_DETECTION_COUNT", True)
