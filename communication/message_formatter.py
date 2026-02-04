"""
Message Formatter Module
MCU로 전송할 메시지를 포맷팅하는 모듈
"""

from typing import List
from tracking.anomaly_detector import AnomalyResult


class MessageFormatter:
    """
    UART를 통해 MCU로 전송할 메시지를 포맷팅하는 클래스
    
    메시지 형식: <BOX_ID:ANOMALY_TYPE:MOVEMENT:ASPECT_RATIO_CHANGE>
    예시: <1:MOVEMENT:45.3:0.02>
    """
    
    @staticmethod
    def format_anomaly(result: AnomalyResult) -> str:
        """
        이상 감지 결과를 UART 메시지 형식으로 변환
        
        Args:
            result: 이상 감지 결과
        
        Returns:
            포맷된 메시지 문자열
        """
        message = (f"<{result.box_id}:"
                  f"{result.anomaly_type.value}:"
                  f"{result.movement_value:.2f}:"
                  f"{result.aspect_ratio_change:.3f}>")
        
        return message
    
    @staticmethod
    def format_multiple(results: List[AnomalyResult]) -> str:
        """
        여러 박스의 이상 감지 결과를 하나의 메시지로 변환
        
        Args:
            results: 이상 감지 결과 리스트
        
        Returns:
            포맷된 메시지 문자열 (개행으로 구분)
        """
        if not results:
            return ""
        
        messages = [MessageFormatter.format_anomaly(r) for r in results]
        return "\n".join(messages)
    
    @staticmethod
    def format_system_status(status: str, fps: float = 0.0, box_count: int = 0) -> str:
        """
        시스템 상태 메시지 포맷팅
        
        Args:
            status: 상태 메시지
            fps: 현재 FPS
            box_count: 검출된 박스 개수
        
        Returns:
            포맷된 상태 메시지
        """
        message = f"<SYS:{status}:FPS={fps:.1f}:BOXES={box_count}>"
        return message


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    """Message Formatter 단위 테스트"""
    from tracking.anomaly_detector import AnomalyResult, AnomalyType
    
    # 테스트용 결과 생성
    result1 = AnomalyResult(
        box_id=1,
        anomaly_type=AnomalyType.NORMAL,
        movement_value=5.2,
        aspect_ratio_change=0.01
    )
    
    result2 = AnomalyResult(
        box_id=2,
        anomaly_type=AnomalyType.MOVEMENT,
        movement_value=45.3,
        aspect_ratio_change=0.02
    )
    
    result3 = AnomalyResult(
        box_id=3,
        anomaly_type=AnomalyType.BOTH,
        movement_value=52.1,
        aspect_ratio_change=0.25
    )
    
    # 단일 메시지 포맷팅
    print("=== Single Message ===")
    print(MessageFormatter.format_anomaly(result1))
    print(MessageFormatter.format_anomaly(result2))
    print(MessageFormatter.format_anomaly(result3))
    
    # 다중 메시지 포맷팅
    print("\n=== Multiple Messages ===")
    results = [result1, result2, result3]
    print(MessageFormatter.format_multiple(results))
    
    # 시스템 상태 메시지
    print("\n=== System Status ===")
    print(MessageFormatter.format_system_status("RUNNING", fps=25.5, box_count=3))
