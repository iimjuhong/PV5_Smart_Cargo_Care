"""
Logger Module
로깅 유틸리티 모듈
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from config.settings import (
    LOG_LEVEL,
    LOG_TO_FILE,
    LOG_FILE_PATH,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT
)


def setup_logger(name: str = "SmartCargoCare") -> logging.Logger:
    """
    로거 설정 및 반환

    Args:
        name: 로거 이름

    Returns:
        설정된 Logger 객체
    """
    logger = logging.getLogger(name)

    # 로그 레벨 설정
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    # 기존 핸들러 제거 (중복 방지)
    logger.handlers.clear()

    # 포맷터 설정
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (RotatingFileHandler 사용)
    if LOG_TO_FILE:
        # 로그 디렉토리 생성
        log_dir = os.path.dirname(LOG_FILE_PATH)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 전역 로거 인스턴스
logger = setup_logger()


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    """Logger 테스트"""
    test_logger = setup_logger("TestLogger")
    
    test_logger.debug("This is a DEBUG message")
    test_logger.info("This is an INFO message")
    test_logger.warning("This is a WARNING message")
    test_logger.error("This is an ERROR message")
    test_logger.critical("This is a CRITICAL message")
    
    print(f"\nLog level: {LOG_LEVEL}")
    print(f"Log to file: {LOG_TO_FILE}")
    if LOG_TO_FILE:
        print(f"Log file path: {LOG_FILE_PATH}")
