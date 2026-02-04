"""
Box Information Data Class
박스 정보를 저장하는 데이터 클래스
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import time


@dataclass
class BoxInfo:
    """
    개별 박스의 정보를 담는 데이터 클래스

    Attributes:
        id (int): 박스 고유 ID
        bbox (Tuple[int, int, int, int]): 바운딩 박스 좌표 (x1, y1, x2, y2)
        center (Tuple[float, float]): 중심점 좌표 (cx, cy)
        aspect_ratio (float): 종횡비 (width / height)
        confidence (float): 검출 신뢰도 (0.0 ~ 1.0)
        class_id (int): 객체 클래스 ID
        timestamp (float): 검출 시간 (Unix timestamp)
    """

    id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[float, float]  # (cx, cy)
    aspect_ratio: float
    confidence: float
    class_id: int
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        """타임스탬프가 없으면 현재 시간으로 설정"""
        if self.timestamp is None:
            self.timestamp = time.time()
    
    @property
    def width(self) -> int:
        """바운딩 박스의 너비"""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        """바운딩 박스의 높이"""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> int:
        """바운딩 박스의 면적"""
        return self.width * self.height
    
    def get_center(self) -> Tuple[float, float]:
        """중심점 좌표 반환"""
        return self.center
    
    def __repr__(self) -> str:
        """박스 정보를 문자열로 표현"""
        return (f"BoxInfo(id={self.id}, "
                f"center=({self.center[0]:.1f}, {self.center[1]:.1f}), "
                f"size={self.width}x{self.height}, "
                f"aspect_ratio={self.aspect_ratio:.2f}, "
                f"confidence={self.confidence:.2f})")


def calculate_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """
    바운딩 박스로부터 중심점 계산
    
    Args:
        bbox: (x1, y1, x2, y2) 형식의 바운딩 박스
    
    Returns:
        (cx, cy) 중심점 좌표
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx, cy)


def calculate_aspect_ratio(bbox: Tuple[int, int, int, int]) -> float:
    """
    바운딩 박스의 종횡비 계산

    Args:
        bbox: (x1, y1, x2, y2) 형식의 바운딩 박스

    Returns:
        종횡비 (width / height), 유효하지 않은 경우 0.0 반환
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    if width == 0 or height == 0:
        return 0.0

    return width / height
