"""
Box Tracker Module
프레임 간 박스 매칭 및 ID 추적 모듈
"""

import logging
from dataclasses import replace
from typing import List, Dict, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from detection.box_info import BoxInfo
from config.settings import TRACKING_IOU_THRESHOLD, MAX_DISAPPEARED_FRAMES

logger = logging.getLogger(__name__)


class BoxTracker:
    """
    IoU 기반 박스 추적 클래스
    
    프레임 간 박스를 매칭하고 고유 ID를 할당합니다.
    """
    
    def __init__(self, iou_threshold: float = TRACKING_IOU_THRESHOLD):
        """
        박스 추적기 초기화
        
        Args:
            iou_threshold: IoU 임계값 (매칭 기준)
        """
        self.iou_threshold = iou_threshold
        self.next_id = 0  # 다음에 할당할 ID
        
        # 추적 중인 박스 정보 저장
        self.tracked_boxes: Dict[int, BoxInfo] = {}  # {box_id: BoxInfo}
        self.disappeared_counts: Dict[int, int] = {}  # {box_id: disappeared_count}
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """
        두 바운딩 박스의 IoU (Intersection over Union) 계산
        
        Args:
            bbox1: (x1, y1, x2, y2) 형식의 첫 번째 박스
            bbox2: (x1, y1, x2, y2) 형식의 두 번째 박스
        
        Returns:
            IoU 값 (0.0 ~ 1.0)
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # 교집합 영역 계산
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        # 교집합 면적
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height
        
        # 각 박스의 면적
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        # 합집합 면적
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        # IoU 계산
        iou = inter_area / union_area
        return iou
    
    def update(self, detected_boxes: List[BoxInfo]) -> List[BoxInfo]:
        """
        검출된 박스로 추적 정보 업데이트

        Args:
            detected_boxes: 현재 프레임에서 검출된 박스 리스트

        Returns:
            ID가 할당된 새 박스 리스트 (원본 객체를 수정하지 않음)
        """
        result_boxes: List[BoxInfo] = []

        # 추적 중인 박스가 없으면 모든 박스에 새 ID 할당
        if len(self.tracked_boxes) == 0:
            for box in detected_boxes:
                new_box = replace(box, id=self.next_id)
                self.tracked_boxes[self.next_id] = new_box
                self.disappeared_counts[self.next_id] = 0
                result_boxes.append(new_box)
                self.next_id += 1
            return result_boxes

        # 검출된 박스가 없으면 모든 추적 박스의 disappeared count 증가
        if len(detected_boxes) == 0:
            for box_id in list(self.disappeared_counts.keys()):
                self.disappeared_counts[box_id] += 1

                # 일정 프레임 이상 사라진 박스는 제거
                if self.disappeared_counts[box_id] > MAX_DISAPPEARED_FRAMES:
                    del self.tracked_boxes[box_id]
                    del self.disappeared_counts[box_id]

            return []

        # IoU 매칭 수행
        tracked_ids = list(self.tracked_boxes.keys())
        num_tracked = len(tracked_ids)
        num_detected = len(detected_boxes)

        # IoU 행렬 계산 (비용 행렬로 변환: 1 - IoU)
        cost_matrix = np.zeros((num_tracked, num_detected))

        for i, tracked_id in enumerate(tracked_ids):
            tracked_bbox = self.tracked_boxes[tracked_id].bbox
            for j, detected_box in enumerate(detected_boxes):
                detected_bbox = detected_box.bbox
                iou = self.calculate_iou(tracked_bbox, detected_bbox)
                cost_matrix[i, j] = 1.0 - iou

        # Hungarian Algorithm을 사용한 최적 매칭
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_pairs: List[Tuple[int, int]] = []  # (tracked_id, detected_idx)
        unmatched_tracked = set(range(num_tracked))
        unmatched_detected = set(range(num_detected))

        for row_idx, col_idx in zip(row_indices, col_indices):
            iou = 1.0 - cost_matrix[row_idx, col_idx]
            if iou >= self.iou_threshold:
                matched_pairs.append((tracked_ids[row_idx], col_idx))
                unmatched_tracked.discard(row_idx)
                unmatched_detected.discard(col_idx)

        # 매칭된 박스 ID 할당
        for tracked_id, detected_idx in matched_pairs:
            new_box = replace(detected_boxes[detected_idx], id=tracked_id)
            self.tracked_boxes[tracked_id] = new_box
            self.disappeared_counts[tracked_id] = 0
            result_boxes.append(new_box)

        # 매칭되지 않은 추적 박스 처리
        for i in unmatched_tracked:
            tracked_id = tracked_ids[i]
            self.disappeared_counts[tracked_id] += 1

            # 일정 프레임 이상 사라진 박스는 제거
            if self.disappeared_counts[tracked_id] > MAX_DISAPPEARED_FRAMES:
                del self.tracked_boxes[tracked_id]
                del self.disappeared_counts[tracked_id]

        # 매칭되지 않은 검출 박스에 새 ID 할당
        for j in unmatched_detected:
            new_box = replace(detected_boxes[j], id=self.next_id)
            self.tracked_boxes[self.next_id] = new_box
            self.disappeared_counts[self.next_id] = 0
            result_boxes.append(new_box)
            self.next_id += 1

        return result_boxes
    
    def get_tracked_count(self) -> int:
        """현재 추적 중인 박스 개수 반환"""
        return len(self.tracked_boxes)
    
    def reset(self):
        """추적 정보 초기화"""
        self.tracked_boxes.clear()
        self.disappeared_counts.clear()
        self.next_id = 0


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    """Box Tracker 단위 테스트"""
    import logging
    logging.basicConfig(level=logging.INFO)

    tracker = BoxTracker()

    # 테스트용 박스 생성 (프레임 1)
    frame1_boxes = [
        BoxInfo(id=-1, bbox=(100, 100, 200, 200), center=(150, 150),
                aspect_ratio=1.0, confidence=0.9, class_id=0),
        BoxInfo(id=-1, bbox=(300, 150, 400, 250), center=(350, 200),
                aspect_ratio=1.0, confidence=0.85, class_id=0),
    ]

    # 첫 프레임 추적
    tracked_boxes = tracker.update(frame1_boxes)
    logger.info("Frame 1:")
    for box in tracked_boxes:
        logger.info(f"  Box ID: {box.id}, Center: {box.center}")

    # 테스트용 박스 생성 (프레임 2 - 약간 이동)
    frame2_boxes = [
        BoxInfo(id=-1, bbox=(105, 105, 205, 205), center=(155, 155),
                aspect_ratio=1.0, confidence=0.9, class_id=0),
        BoxInfo(id=-1, bbox=(305, 155, 405, 255), center=(355, 205),
                aspect_ratio=1.0, confidence=0.85, class_id=0),
    ]

    # 두 번째 프레임 추적
    tracked_boxes = tracker.update(frame2_boxes)
    logger.info("Frame 2:")
    for box in tracked_boxes:
        logger.info(f"  Box ID: {box.id}, Center: {box.center}")

    logger.info(f"Total tracked boxes: {tracker.get_tracked_count()}")
