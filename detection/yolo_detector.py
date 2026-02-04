"""
YOLO Detector Module
YOLOv8 모델을 사용한 객체 검출 모듈
"""

import logging
import os
import shutil
from typing import List, Optional, Dict
import cv2
import numpy as np
from ultralytics import YOLO

from detection.box_info import BoxInfo, calculate_center, calculate_aspect_ratio
from config.settings import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    AUTO_DOWNLOAD_MODEL
)

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    YOLOv8 기반 객체 검출기
    
    박스(또는 지정된 클래스)를 검출하고 바운딩 박스 정보를 반환합니다.
    """
    
    def __init__(self, model_path: str = MODEL_PATH, confidence: float = CONFIDENCE_THRESHOLD):
        """
        YOLO 검출기 초기화
        
        Args:
            model_path: YOLO 모델 파일 경로 (.pt 파일)
            confidence: 검출 신뢰도 임계값 (0.0 ~ 1.0)
        """
        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """YOLO 모델 로딩"""
        # 모델 파일이 존재하는지 확인
        if os.path.exists(self.model_path):
            try:
                logger.info(f"Loading YOLO model from: {self.model_path}")
                self.model = YOLO(self.model_path)
                logger.info("YOLO model loaded successfully")
                return
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")

        # 모델 파일이 없거나 로드 실패 시 자동 다운로드
        if AUTO_DOWNLOAD_MODEL:
            logger.info("Model not found. Downloading YOLOv8n model...")
            try:
                # ultralytics가 자동으로 다운로드
                self.model = YOLO("yolov8n.pt")
                logger.info("Model downloaded successfully")

                # 다운로드된 모델을 models/ 폴더에 복사
                self._save_model_to_path()
            except Exception as download_error:
                logger.error(f"Failed to download model: {download_error}")
                raise RuntimeError(
                    f"Cannot load YOLO model from '{self.model_path}' "
                    f"and auto-download also failed: {download_error}"
                ) from download_error
        else:
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                f"Set AUTO_DOWNLOAD_MODEL=True to enable auto-download."
            )

    def _save_model_to_path(self) -> None:
        """다운로드된 모델을 지정된 경로에 저장"""
        try:
            # models 디렉토리 생성
            model_dir = os.path.dirname(self.model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)
                logger.info(f"Created directory: {model_dir}")

            # 현재 디렉토리에 다운로드된 모델 확인
            downloaded_model = "yolov8n.pt"
            if os.path.exists(downloaded_model) and self.model_path != downloaded_model:
                shutil.copy2(downloaded_model, self.model_path)
                os.remove(downloaded_model)  # 원본 삭제
                logger.info(f"Model saved to: {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not save model to {self.model_path}: {e}")
    
    def detect(self, frame: np.ndarray, target_classes: Optional[List[int]] = None) -> List[BoxInfo]:
        """
        프레임에서 객체 검출

        Args:
            frame: 입력 이미지 (BGR 형식)
            target_classes: 검출할 클래스 ID 리스트 (None이면 모든 클래스)

        Returns:
            검출된 박스 정보 리스트 (BoxInfo 객체들)
        """
        if self.model is None:
            logger.warning("Model not loaded. Returning empty detection.")
            return []
        
        # YOLO 추론 수행
        results = self.model(frame, conf=self.confidence, iou=IOU_THRESHOLD, verbose=False)
        
        boxes_info = []
        
        # 검출 결과 파싱
        for result in results:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # 바운딩 박스 좌표 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                # 신뢰도 및 클래스 ID
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # 타겟 클래스 필터링
                if target_classes is not None and class_id not in target_classes:
                    continue
                
                # 중심점 및 종횡비 계산
                center = calculate_center(bbox)
                aspect_ratio = calculate_aspect_ratio(bbox)
                
                # BoxInfo 객체 생성
                box_info = BoxInfo(
                    id=-1,  # ID는 추적 단계에서 할당됨
                    bbox=bbox,
                    center=center,
                    aspect_ratio=aspect_ratio,
                    confidence=confidence,
                    class_id=class_id
                )
                
                boxes_info.append(box_info)
        
        return boxes_info
    
    def get_class_names(self) -> Dict[int, str]:
        """
        YOLO 모델의 클래스 이름 딕셔너리 반환
        
        Returns:
            {class_id: class_name} 형식의 딕셔너리
        """
        if self.model is None:
            return {}
        
        return self.model.names
    
    def __del__(self) -> None:
        """소멸자: 리소스 정리"""
        if self.model is not None:
            del self.model


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    """
    YOLO 검출기 단위 테스트
    웹캠에서 프레임을 캡처하여 객체를 검출합니다.
    """
    logging.basicConfig(level=logging.INFO)

    detector = YOLODetector()

    # 웹캠 열기
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Cannot open camera")
        exit()

    logger.info("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Cannot read frame")
            break

        # 객체 검출
        boxes = detector.detect(frame)

        # 결과 시각화
        for box in boxes:
            x1, y1, x2, y2 = box.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 라벨 표시
            label = f"Class {box.class_id}: {box.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 검출 개수 표시
        cv2.putText(frame, f"Detected: {len(boxes)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO Detection Test", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
