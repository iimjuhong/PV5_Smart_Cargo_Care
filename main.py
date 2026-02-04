"""
Smart Cargo Care - Main Application
메인 실행 파일: 모든 모듈을 통합하여 실시간 화물 모니터링 시스템 실행
"""

import cv2
import time
from typing import List

# 모듈 임포트
from camera.video_stream import VideoStream
from detection.yolo_detector import YOLODetector
from tracking.box_tracker import BoxTracker
from tracking.anomaly_detector import AnomalyDetector, AnomalyType
from communication.uart_handler import UARTHandler
from communication.message_formatter import MessageFormatter
from utils.visualization import Visualizer
from utils.logger import logger

# 설정 임포트
from config.settings import (
    ENABLE_VISUALIZATION,
    WINDOW_NAME,
    SHOW_FPS,
    SHOW_DETECTION_COUNT
)


class SmartCargoCareSys:
    """
    Smart Cargo Care 시스템 메인 클래스
    
    전체 파이프라인을 통합하여 실행합니다.
    """
    
    def __init__(self):
        """시스템 초기화"""
        logger.info("=" * 60)
        logger.info("Smart Cargo Care System Initializing...")
        logger.info("=" * 60)
        
        # 모듈 초기화
        self.video_stream = None
        self.detector = None
        self.tracker = None
        self.anomaly_detector = None
        self.uart = None
        self.visualizer = None
        
        # 성능 측정 변수
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # 시스템 상태
        self.is_running = False
    
    def initialize_modules(self) -> bool:
        """
        모든 모듈 초기화
        
        Returns:
            초기화 성공 여부
        """
        try:
            # 1. 비디오 스트림 초기화
            logger.info("Initializing video stream...")
            self.video_stream = VideoStream()
            if not self.video_stream.is_opened:
                logger.error("Failed to open video stream")
                return False
            
            # 2. YOLO 검출기 초기화
            logger.info("Loading YOLO model...")
            self.detector = YOLODetector()
            
            # 3. 박스 추적기 초기화
            logger.info("Initializing box tracker...")
            self.tracker = BoxTracker()
            
            # 4. 이상 감지기 초기화
            logger.info("Initializing anomaly detector...")
            self.anomaly_detector = AnomalyDetector()
            
            # 5. UART 통신 초기화
            logger.info("Initializing UART communication...")
            self.uart = UARTHandler()
            
            # 6. 시각화 도구 초기화
            if ENABLE_VISUALIZATION:
                logger.info("Initializing visualizer...")
                self.visualizer = Visualizer()
            
            logger.info("All modules initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}")
            return False
    
    def process_frame(self, frame):
        """
        단일 프레임 처리 파이프라인
        
        Args:
            frame: 입력 프레임 (BGR)
        
        Returns:
            처리된 프레임 (시각화된 경우)
        """
        # 1. YOLO 객체 검출
        detected_boxes = self.detector.detect(frame)
        
        # 2. 박스 추적 (ID 할당)
        tracked_boxes = self.tracker.update(detected_boxes)
        
        # 3. 이상 감지
        anomaly_results = self.anomaly_detector.detect(tracked_boxes)
        
        # 4. UART 메시지 전송
        for result in anomaly_results:
            # 이상이 감지된 경우에만 전송 (옵션: 모든 상태 전송)
            if result.anomaly_type != AnomalyType.NORMAL:
                message = MessageFormatter.format_anomaly(result)
                self.uart.send(message)
                logger.warning(f"Anomaly detected: {result}")
        
        # 5. 시각화 (활성화된 경우)
        output_frame = frame
        if ENABLE_VISUALIZATION and self.visualizer:
            output_frame = self.visualizer.draw_boxes(
                frame, tracked_boxes, anomaly_results
            )
            
            # 이상 개수 계산
            anomaly_count = sum(
                1 for r in anomaly_results 
                if r.anomaly_type != AnomalyType.NORMAL
            )
            
            if SHOW_FPS or SHOW_DETECTION_COUNT:
                output_frame = self.visualizer.draw_info(
                    output_frame,
                    fps=self.fps,
                    box_count=len(tracked_boxes),
                    anomaly_count=anomaly_count
                )
        
        return output_frame
    
    def update_fps(self):
        """FPS 계산 및 업데이트"""
        self.frame_count += 1
        elapsed_time = time.time() - self.fps_start_time
        
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.fps_start_time = time.time()
    
    def run(self):
        """메인 루프 실행"""
        # 모듈 초기화
        if not self.initialize_modules():
            logger.error("Failed to initialize. Exiting...")
            return
        
        logger.info("=" * 60)
        logger.info("System started successfully!")
        logger.info("Press 'q' to quit")
        logger.info("=" * 60)
        
        self.is_running = True
        
        try:
            while self.is_running:
                # 프레임 읽기
                ret, frame = self.video_stream.read()
                
                if not ret:
                    logger.warning("Failed to read frame. Retrying...")
                    time.sleep(0.1)
                    continue
                
                # 프레임 처리
                output_frame = self.process_frame(frame)
                
                # FPS 업데이트
                self.update_fps()
                
                # 시각화 표시
                if ENABLE_VISUALIZATION:
                    cv2.imshow(WINDOW_NAME, output_frame)
                    
                    # 키 입력 처리
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User requested quit")
                        self.is_running = False
                    elif key == ord('r'):
                        # 'r' 키로 추적 초기화
                        logger.info("Resetting tracker...")
                        self.tracker.reset()
                        self.anomaly_detector.reset()
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        logger.info("Cleaning up resources...")
        
        if self.video_stream:
            self.video_stream.release()
        
        if self.uart:
            self.uart.close()
        
        if ENABLE_VISUALIZATION:
            cv2.destroyAllWindows()
        
        logger.info("System shutdown complete")
        logger.info("=" * 60)


# ==================== 메인 실행 ====================
def main():
    """메인 함수"""
    # 시스템 생성 및 실행
    system = SmartCargoCareSys()
    system.run()


if __name__ == "__main__":
    main()
