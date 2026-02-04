"""
Movement Analyzer Module
박스의 중심점 이동을 분석하는 모듈
"""

import math
from typing import Tuple
from detection.box_info import BoxInfo


class MovementAnalyzer:
    """
    박스의 중심점 변화량을 계산하는 클래스
    """
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """
        두 점 사이의 유클리드 거리 계산
        
        Args:
            point1: (x1, y1) 좌표
            point2: (x2, y2) 좌표
        
        Returns:
            거리 (픽셀 단위)
        """
        x1, y1 = point1
        x2, y2 = point2
        
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance
    
    @staticmethod
    def calculate_movement(prev_box: BoxInfo, curr_box: BoxInfo) -> float:
        """
        이전 박스와 현재 박스의 중심점 이동 거리 계산
        
        Args:
            prev_box: 이전 프레임의 박스 정보
            curr_box: 현재 프레임의 박스 정보
        
        Returns:
            중심점 이동 거리 (픽셀)
        """
        prev_center = prev_box.center
        curr_center = curr_box.center
        
        return MovementAnalyzer.calculate_distance(prev_center, curr_center)
    
    @staticmethod
    def calculate_velocity(prev_box: BoxInfo, curr_box: BoxInfo) -> float:
        """
        박스의 이동 속도 계산 (픽셀/프레임)
        
        Args:
            prev_box: 이전 프레임의 박스 정보
            curr_box: 현재 프레임의 박스 정보
        
        Returns:
            이동 속도 (픽셀/프레임)
        """
        movement = MovementAnalyzer.calculate_movement(prev_box, curr_box)
        
        # 시간 차이 계산 (초 단위)
        time_diff = curr_box.timestamp - prev_box.timestamp
        
        if time_diff == 0:
            return 0.0
        
        # 속도 = 거리 / 시간
        velocity = movement / time_diff
        return velocity
    
    @staticmethod
    def calculate_direction(prev_box: BoxInfo, curr_box: BoxInfo) -> Tuple[float, float]:
        """
        박스의 이동 방향 벡터 계산
        
        Args:
            prev_box: 이전 프레임의 박스 정보
            curr_box: 현재 프레임의 박스 정보
        
        Returns:
            (dx, dy) 방향 벡터
        """
        prev_center = prev_box.center
        curr_center = curr_box.center
        
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        
        return (dx, dy)


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    """Movement Analyzer 단위 테스트"""
    from detection.box_info import BoxInfo
    import time
    
    # 테스트용 박스 생성
    box1 = BoxInfo(
        id=1,
        bbox=(100, 100, 200, 200),
        center=(150.0, 150.0),
        aspect_ratio=1.0,
        confidence=0.9,
        class_id=0,
        timestamp=time.time()
    )
    
    # 0.1초 대기
    time.sleep(0.1)
    
    # 이동한 박스
    box2 = BoxInfo(
        id=1,
        bbox=(150, 120, 250, 220),
        center=(200.0, 170.0),
        aspect_ratio=1.0,
        confidence=0.9,
        class_id=0,
        timestamp=time.time()
    )
    
    # 이동 거리 계산
    movement = MovementAnalyzer.calculate_movement(box1, box2)
    print(f"Movement distance: {movement:.2f} pixels")
    
    # 이동 속도 계산
    velocity = MovementAnalyzer.calculate_velocity(box1, box2)
    print(f"Velocity: {velocity:.2f} pixels/second")
    
    # 이동 방향 계산
    direction = MovementAnalyzer.calculate_direction(box1, box2)
    print(f"Direction: dx={direction[0]:.2f}, dy={direction[1]:.2f}")
