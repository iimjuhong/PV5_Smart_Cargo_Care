"""
Video Stream Module
비디오 스트림(웹캠 또는 IP 카메라)을 캡처하는 모듈
"""

import logging
import cv2
import numpy as np
from typing import Optional, Tuple, Union

from config.settings import (
    CAMERA_SOURCE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS_LIMIT
)

logger = logging.getLogger(__name__)


class VideoStream:
    """
    비디오 스트림 캡처 클래스
    
    웹캠, USB 카메라, 또는 IP 카메라(스마트폰)에서 영상을 캡처합니다.
    """
    
    def __init__(self, 
                 source: Union[int, str] = CAMERA_SOURCE,
                 width: int = FRAME_WIDTH,
                 height: int = FRAME_HEIGHT,
                 fps_limit: int = FPS_LIMIT):
        """
        비디오 스트림 초기화
        
        Args:
            source: 카메라 소스
                - int: 웹캠 인덱스 (0, 1, 2, ...)
                - str: IP 카메라 URL (예: "http://192.168.0.10:8080/video")
            width: 프레임 너비
            height: 프레임 높이
            fps_limit: FPS 제한 (0 = 제한 없음)
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps_limit = fps_limit
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        
        self._open()
    
    def _open(self) -> bool:
        """
        비디오 캡처 디바이스 열기

        Returns:
            열기 성공 여부
        """
        try:
            logger.info(f"Opening camera source: {self.source}")
            self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False

            # 해상도 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # FPS 제한 설정
            if self.fps_limit > 0:
                self.cap.set(cv2.CAP_PROP_FPS, self.fps_limit)

            # 실제 설정된 값 확인
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            logger.info("Camera opened successfully")
            logger.info(f"  Resolution: {actual_width}x{actual_height}")
            logger.info(f"  FPS: {actual_fps:.1f}")

            self.is_opened = True
            return True

        except Exception as e:
            logger.error(f"Exception while opening camera: {e}")
            logger.info("Troubleshooting tips:")
            logger.info("  - For webcam: Try source=0, 1, or 2")
            logger.info("  - For IP camera: Check URL and ensure app is running")
            logger.info("  - For DroidCam: URL format is usually http://IP:4747/video")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        프레임 읽기

        Returns:
            (성공 여부, 프레임 이미지)
        """
        if not self.is_opened or self.cap is None:
            return False, None

        try:
            ret, frame = self.cap.read()

            if not ret:
                logger.warning("Failed to read frame")
                return False, None

            return True, frame

        except Exception as e:
            logger.error(f"Exception while reading frame: {e}")
            return False, None
    
    def get_fps(self) -> float:
        """
        현재 설정된 FPS 반환
        
        Returns:
            FPS 값
        """
        if self.cap is None:
            return 0.0
        
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def get_resolution(self) -> Tuple[int, int]:
        """
        현재 설정된 해상도 반환
        
        Returns:
            (width, height)
        """
        if self.cap is None:
            return (0, 0)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return (width, height)
    
    def release(self):
        """비디오 캡처 디바이스 해제"""
        if self.cap is not None and self.is_opened:
            try:
                self.cap.release()
                logger.info("Camera released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            finally:
                self.is_opened = False

    def reconnect(self) -> bool:
        """
        카메라 재연결 시도

        Returns:
            재연결 성공 여부
        """
        logger.info("Attempting to reconnect camera...")
        self.release()
        return self._open()

    def __enter__(self) -> "VideoStream":
        """컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """컨텍스트 매니저 종료"""
        self.release()

    def __del__(self):
        """소멸자: 리소스 정리"""
        self.release()


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    """Video Stream 단위 테스트"""
    import time
    
    print("=== Video Stream Test ===")
    print("Testing camera stream...")
    print("Press 'q' to quit\n")
    
    # 웹캠 테스트 (source=0)
    stream = VideoStream(source=0)
    
    if not stream.is_opened:
        print("[ERROR] Camera test failed")
        exit()
    
    # FPS 계산을 위한 변수
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0
    
    while True:
        # 프레임 읽기
        ret, frame = stream.read()
        
        if not ret:
            print("[ERROR] Cannot read frame")
            break
        
        # FPS 계산
        fps_frame_count += 1
        elapsed_time = time.time() - fps_start_time
        
        if elapsed_time >= 1.0:
            current_fps = fps_frame_count / elapsed_time
            fps_frame_count = 0
            fps_start_time = time.time()
        
        # 프레임에 FPS 표시
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 해상도 표시
        height, width = frame.shape[:2]
        cv2.putText(frame, f"Resolution: {width}x{height}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 프레임 표시
        cv2.imshow("Video Stream Test", frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 리소스 정리
    stream.release()
    cv2.destroyAllWindows()
    
    print("\nTest completed")
