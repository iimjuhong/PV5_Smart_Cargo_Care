"""
UART Handler Module
MCU와의 UART 통신을 처리하는 모듈
"""

import logging
import serial
from typing import Optional
from config.settings import UART_PORT, UART_BAUDRATE, UART_TIMEOUT, UART_ENABLED

logger = logging.getLogger(__name__)


class UARTHandler:
    """
    UART 시리얼 통신 핸들러
    
    MCU로 메시지를 전송하고 연결 상태를 관리합니다.
    """
    
    def __init__(self, 
                 port: str = UART_PORT,
                 baudrate: int = UART_BAUDRATE,
                 timeout: float = UART_TIMEOUT,
                 enabled: bool = UART_ENABLED):
        """
        UART 핸들러 초기화
        
        Args:
            port: 시리얼 포트 이름 (예: "/dev/cu.usbserial-1420", "COM3")
            baudrate: Baud rate (예: 9600, 115200)
            timeout: 읽기 타임아웃 (초)
            enabled: UART 활성화 여부
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.enabled = enabled
        
        self.serial: Optional[serial.Serial] = None
        self.is_connected = False
        
        if self.enabled:
            self.connect()
    
    def connect(self) -> bool:
        """
        UART 연결 시도

        Returns:
            연결 성공 여부
        """
        if not self.enabled:
            logger.info("UART is disabled in settings")
            return False

        try:
            logger.info(f"Connecting to UART port: {self.port}")
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self.is_connected = True
            logger.info(f"UART connected successfully at {self.baudrate} baud")
            return True

        except serial.SerialException as e:
            logger.error(f"Failed to connect to UART: {e}")
            logger.info("Please check:")
            logger.info(f"  1. Port name is correct: {self.port}")
            logger.info("  2. MCU is connected")
            logger.info(f"  3. Port permissions (run: sudo chmod 660 {self.port})")
            self.is_connected = False
            return False
    
    def send(self, message: str) -> bool:
        """
        UART를 통해 메시지 전송

        Args:
            message: 전송할 메시지 문자열

        Returns:
            전송 성공 여부
        """
        if not self.enabled:
            # UART가 비활성화되어 있으면 콘솔에만 출력
            logger.debug(f"[UART-DISABLED] Would send: {message}")
            return True

        if not self.is_connected or self.serial is None:
            logger.warning("UART not connected. Message not sent.")
            return False

        try:
            # 메시지 끝에 개행 문자 추가
            message_bytes = (message + "\n").encode('utf-8')
            self.serial.write(message_bytes)
            self.serial.flush()  # 버퍼 비우기

            # 디버깅 출력
            logger.debug(f"[UART-SENT] {message}")
            return True

        except serial.SerialException as e:
            logger.error(f"Failed to send message: {e}")
            self.is_connected = False
            return False
    
    def read(self, num_bytes: int = 1) -> Optional[bytes]:
        """
        UART로부터 데이터 읽기

        Args:
            num_bytes: 읽을 바이트 수

        Returns:
            읽은 데이터 (bytes) 또는 None
        """
        if not self.is_connected or self.serial is None:
            return None

        try:
            data = self.serial.read(num_bytes)
            return data if data else None

        except serial.SerialException as e:
            logger.error(f"Failed to read from UART: {e}")
            self.is_connected = False
            return None
    
    def readline(self) -> Optional[str]:
        """
        UART로부터 한 줄 읽기 (개행 문자까지)

        Returns:
            읽은 문자열 또는 None
        """
        if not self.is_connected or self.serial is None:
            return None

        try:
            line = self.serial.readline().decode('utf-8').strip()
            return line if line else None

        except serial.SerialException as e:
            logger.error(f"Failed to read line from UART: {e}")
            self.is_connected = False
            return None
    
    def close(self):
        """UART 연결 종료"""
        if self.serial is not None and self.is_connected:
            try:
                self.serial.close()
                logger.info("UART connection closed")
            except Exception as e:
                logger.error(f"Error closing UART: {e}")
            finally:
                self.is_connected = False
    
    def __del__(self):
        """소멸자: UART 연결 정리"""
        self.close()


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    """UART Handler 단위 테스트"""
    import time
    logging.basicConfig(level=logging.DEBUG)

    # 테스트 모드 (UART 비활성화)
    logger.info("=== Test Mode (UART Disabled) ===")
    uart = UARTHandler(enabled=False)

    # 메시지 전송 테스트
    uart.send("<1:NORMAL:5.2:0.01>")
    uart.send("<2:MOVEMENT:45.3:0.02>")

    time.sleep(1)

    # 실제 UART 테스트 (포트가 연결되어 있을 때만 동작)
    logger.info("=== Real UART Test ===")
    logger.info("Attempting to connect to real UART...")
    logger.info("(This will fail if no MCU is connected)")

    uart_real = UARTHandler(
        port="/dev/cu.usbserial-1420",  # 실제 포트로 변경 필요
        baudrate=9600,
        enabled=True
    )

    if uart_real.is_connected:
        logger.info("Successfully connected!")
        uart_real.send("<TEST:MESSAGE>")
        time.sleep(0.5)
        uart_real.close()
    else:
        logger.info("Connection failed (expected if no MCU)")
