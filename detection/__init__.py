"""
Detection module
객체 검출 관련 클래스와 함수를 제공합니다.
"""

from detection.box_info import BoxInfo, calculate_center, calculate_aspect_ratio
from detection.yolo_detector import YOLODetector

__all__ = [
    "BoxInfo",
    "calculate_center",
    "calculate_aspect_ratio",
    "YOLODetector",
]
