"""
Anomaly Detector Module
박스의 이상 상태를 감지하는 모듈
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
from detection.box_info import BoxInfo
from tracking.movement_analyzer import MovementAnalyzer
from config.settings import MOVEMENT_THRESHOLD, ASPECT_RATIO_THRESHOLD


class AnomalyType(Enum):
    """이상 징후 타입"""
    NORMAL = "NORMAL"           # 정상
    MOVEMENT = "MOVEMENT"       # 이동 감지
    COLLISION = "COLLISION"     # 충돌/변형 감지
    BOTH = "BOTH"               # 이동 + 충돌


@dataclass
class AnomalyResult:
    """
    이상 감지 결과

    Attributes:
        box_id: 박스 ID
        anomaly_type: 이상 징후 타입
        movement_value: 중심점 이동 거리 (픽셀)
        aspect_ratio_change: 종횡비 변화율 (0.0 ~ 1.0)
    """
    box_id: int
    anomaly_type: AnomalyType
    movement_value: float
    aspect_ratio_change: float

    def __repr__(self) -> str:
        return (f"AnomalyResult(id={self.box_id}, "
                f"type={self.anomaly_type.value}, "
                f"movement={self.movement_value:.2f}, "
                f"aspect_change={self.aspect_ratio_change:.3f})")


class AnomalyDetector:
    """
    박스의 이상 상태를 감지하는 클래스
    
    위치 급변(중심점 이동) 및 형태 변형(종횡비 변화)을 감지합니다.
    """
    
    def __init__(self, 
                 movement_threshold: float = MOVEMENT_THRESHOLD,
                 aspect_ratio_threshold: float = ASPECT_RATIO_THRESHOLD):
        """
        이상 감지기 초기화
        
        Args:
            movement_threshold: 이동 거리 임계값 (픽셀)
            aspect_ratio_threshold: 종횡비 변화율 임계값 (0.0 ~ 1.0)
        """
        self.movement_threshold = movement_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        
        # 이전 프레임의 박스 정보 저장
        self.previous_boxes: Dict[int, BoxInfo] = {}  # {box_id: BoxInfo}
    
    def detect(self, current_boxes: List[BoxInfo]) -> List[AnomalyResult]:
        """
        현재 프레임의 박스에서 이상 징후 감지
        
        Args:
            current_boxes: 현재 프레임의 박스 리스트
        
        Returns:
            이상 감지 결과 리스트
        """
        results = []
        
        for curr_box in current_boxes:
            box_id = curr_box.id
            
            # 이전 프레임에 해당 박스가 없으면 정상으로 처리
            if box_id not in self.previous_boxes:
                result = AnomalyResult(
                    box_id=box_id,
                    anomaly_type=AnomalyType.NORMAL,
                    movement_value=0.0,
                    aspect_ratio_change=0.0
                )
                results.append(result)
                continue
            
            prev_box = self.previous_boxes[box_id]
            
            # 1. 중심점 이동 거리 계산
            movement = MovementAnalyzer.calculate_movement(prev_box, curr_box)
            
            # 2. 종횡비 변화율 계산
            prev_ratio = prev_box.aspect_ratio
            curr_ratio = curr_box.aspect_ratio
            
            if prev_ratio == 0:
                aspect_change = 0.0
            else:
                aspect_change = abs(curr_ratio - prev_ratio) / prev_ratio
            
            # 3. 이상 감지 판단
            is_movement_anomaly = movement > self.movement_threshold
            is_collision_anomaly = aspect_change > self.aspect_ratio_threshold
            
            # 이상 타입 결정
            if is_movement_anomaly and is_collision_anomaly:
                anomaly_type = AnomalyType.BOTH
            elif is_movement_anomaly:
                anomaly_type = AnomalyType.MOVEMENT
            elif is_collision_anomaly:
                anomaly_type = AnomalyType.COLLISION
            else:
                anomaly_type = AnomalyType.NORMAL
            
            # 결과 생성
            result = AnomalyResult(
                box_id=box_id,
                anomaly_type=anomaly_type,
                movement_value=movement,
                aspect_ratio_change=aspect_change
            )
            results.append(result)
        
        # 현재 프레임의 박스 정보를 이전 프레임으로 저장 (딕셔너리 컴프리헨션)
        self.previous_boxes = {box.id: box for box in current_boxes}

        return results
    
    def reset(self):
        """이상 감지기 초기화"""
        self.previous_boxes.clear()


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    """Anomaly Detector 단위 테스트"""
    import time
    
    detector = AnomalyDetector(
        movement_threshold=30.0,
        aspect_ratio_threshold=0.15
    )
    
    # 테스트 시나리오 1: 정상 상태
    print("=== Test 1: Normal ===")
    boxes_frame1 = [
        BoxInfo(id=0, bbox=(100, 100, 200, 200), center=(150, 150),
                aspect_ratio=1.0, confidence=0.9, class_id=0)
    ]
    
    results = detector.detect(boxes_frame1)
    for r in results:
        print(r)
    
    time.sleep(0.05)
    
    # 테스트 시나리오 2: 작은 이동 (정상)
    print("\n=== Test 2: Small Movement (Normal) ===")
    boxes_frame2 = [
        BoxInfo(id=0, bbox=(105, 105, 205, 205), center=(155, 155),
                aspect_ratio=1.0, confidence=0.9, class_id=0)
    ]
    
    results = detector.detect(boxes_frame2)
    for r in results:
        print(r)
    
    time.sleep(0.05)
    
    # 테스트 시나리오 3: 큰 이동 (이상)
    print("\n=== Test 3: Large Movement (Anomaly) ===")
    boxes_frame3 = [
        BoxInfo(id=0, bbox=(200, 200, 300, 300), center=(250, 250),
                aspect_ratio=1.0, confidence=0.9, class_id=0)
    ]
    
    results = detector.detect(boxes_frame3)
    for r in results:
        print(r)
    
    time.sleep(0.05)
    
    # 테스트 시나리오 4: 종횡비 변화 (충돌)
    print("\n=== Test 4: Aspect Ratio Change (Collision) ===")
    boxes_frame4 = [
        BoxInfo(id=0, bbox=(200, 200, 320, 300), center=(260, 250),
                aspect_ratio=1.2, confidence=0.9, class_id=0)
    ]
    
    results = detector.detect(boxes_frame4)
    for r in results:
        print(r)
