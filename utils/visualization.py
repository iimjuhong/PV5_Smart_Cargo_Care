"""
Visualization Module
검출 결과를 시각화하는 디버깅 유틸리티
"""

import cv2
import numpy as np
from typing import List

from detection.box_info import BoxInfo
from tracking.anomaly_detector import AnomalyResult, AnomalyType
from config.settings import COLOR_NORMAL, COLOR_MOVEMENT, COLOR_COLLISION


class Visualizer:
    """
    검출 및 추적 결과를 프레임에 시각화하는 클래스
    """
    
    @staticmethod
    def draw_boxes(frame: np.ndarray, 
                   boxes: List[BoxInfo],
                   anomaly_results: List[AnomalyResult] = None) -> np.ndarray:
        """
        프레임에 박스 및 이상 상태 표시
        
        Args:
            frame: 입력 프레임 (BGR)
            boxes: 박스 정보 리스트
            anomaly_results: 이상 감지 결과 리스트 (옵션)
        
        Returns:
            시각화된 프레임
        """
        output = frame.copy()
        
        # 이상 상태를 딕셔너리로 변환 (빠른 검색을 위해)
        anomaly_dict = {}
        if anomaly_results:
            for result in anomaly_results:
                anomaly_dict[result.box_id] = result
        
        for box in boxes:
            x1, y1, x2, y2 = box.bbox
            
            # 이상 상태에 따른 색상 결정
            if box.id in anomaly_dict:
                result = anomaly_dict[box.id]
                if result.anomaly_type == AnomalyType.MOVEMENT:
                    color = COLOR_MOVEMENT
                    label_prefix = "[!] MOVE"
                elif result.anomaly_type == AnomalyType.COLLISION:
                    color = COLOR_COLLISION
                    label_prefix = "[!!] COLLISION"
                elif result.anomaly_type == AnomalyType.BOTH:
                    color = COLOR_COLLISION
                    label_prefix = "[!!] BOTH"
                else:
                    color = COLOR_NORMAL
                    label_prefix = "[OK]"
            else:
                color = COLOR_NORMAL
                label_prefix = "[OK]"
            
            # 바운딩 박스 그리기
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # 중심점 표시
            cx, cy = box.center
            cv2.circle(output, (int(cx), int(cy)), 5, color, -1)
            
            # 라벨 텍스트 구성
            label = f"{label_prefix} ID:{box.id}"
            
            # 이상 정보 추가
            if box.id in anomaly_dict:
                result = anomaly_dict[box.id]
                if result.anomaly_type != AnomalyType.NORMAL:
                    label += f" | M:{result.movement_value:.1f} AR:{result.aspect_ratio_change:.2f}"
            
            # 배경 박스 그리기
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(output, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width + 10, y1),
                         color, -1)
            
            # 라벨 텍스트 그리기
            cv2.putText(output, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output
    
    @staticmethod
    def draw_info(frame: np.ndarray,
                  fps: float,
                  box_count: int,
                  anomaly_count: int = 0) -> np.ndarray:
        """
        프레임에 시스템 정보 표시
        
        Args:
            frame: 입력 프레임
            fps: 현재 FPS
            box_count: 검출된 박스 개수
            anomaly_count: 이상 감지된 박스 개수
        
        Returns:
            정보가 표시된 프레임
        """
        output = frame.copy()
        height, width = output.shape[:2]
        
        # 정보 패널 배경
        panel_height = 100
        cv2.rectangle(output, (0, 0), (300, panel_height), (0, 0, 0), -1)
        cv2.rectangle(output, (0, 0), (300, panel_height), (255, 255, 255), 2)
        
        # FPS 표시
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(output, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 박스 개수 표시
        box_text = f"Boxes: {box_count}"
        cv2.putText(output, box_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 이상 개수 표시
        anomaly_color = (0, 0, 255) if anomaly_count > 0 else (0, 255, 0)
        anomaly_text = f"Anomalies: {anomaly_count}"
        cv2.putText(output, anomaly_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, anomaly_color, 2)
        
        return output
    
    @staticmethod
    def draw_title(frame: np.ndarray, title: str = "Smart Cargo Care") -> np.ndarray:
        """
        프레임에 제목 표시
        
        Args:
            frame: 입력 프레임
            title: 제목 텍스트
        
        Returns:
            제목이 표시된 프레임
        """
        output = frame.copy()
        height, width = output.shape[:2]
        
        # 제목 위치 (상단 중앙)
        (text_width, text_height), _ = cv2.getTextSize(
            title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )
        
        x = (width - text_width) // 2
        y = 30
        
        # 배경 박스
        cv2.rectangle(output, 
                     (x - 10, y - text_height - 10),
                     (x + text_width + 10, y + 10),
                     (0, 0, 0), -1)
        
        # 제목 텍스트
        cv2.putText(output, title, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        return output


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    """Visualizer 테스트"""
    import time
    
    # 테스트용 빈 프레임 생성
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 테스트용 박스 생성
    boxes = [
        BoxInfo(id=0, bbox=(50, 50, 150, 150), center=(100, 100),
                aspect_ratio=1.0, confidence=0.9, class_id=0),
        BoxInfo(id=1, bbox=(200, 100, 350, 250), center=(275, 175),
                aspect_ratio=1.0, confidence=0.85, class_id=0),
    ]
    
    # 테스트용 이상 결과
    anomaly_results = [
        AnomalyResult(box_id=0, anomaly_type=AnomalyType.NORMAL,
                     movement_value=5.2, aspect_ratio_change=0.01),
        AnomalyResult(box_id=1, anomaly_type=AnomalyType.MOVEMENT,
                     movement_value=45.3, aspect_ratio_change=0.02),
    ]
    
    # 시각화
    vis = Visualizer()
    frame = vis.draw_boxes(frame, boxes, anomaly_results)
    frame = vis.draw_info(frame, fps=25.5, box_count=2, anomaly_count=1)
    frame = vis.draw_title(frame)
    
    # 결과 표시
    cv2.imshow("Visualization Test", frame)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
