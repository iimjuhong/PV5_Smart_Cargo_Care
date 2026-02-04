# Smart Cargo Care

물류용 차량의 적재함 상태를 실시간으로 모니터링하고, 화물 박스의 이상 징후(낙하, 충돌, 쏠림)를 감지하여 운전석으로 경고를 전송하는 지능형 화물 관리 시스템입니다.

## 주요 특징

- **실시간 객체 감지**: Hugging Face waste-classification 모델을 사용한 cardboard 박스 검출
- **정밀 추적**: Hungarian Algorithm 기반 프레임 간 박스 ID 매칭
- **지능형 이상 감지**: 위치 급변 및 형태 변형 감지
- **MCU 통신**: UART를 통한 실시간 경고 전송
- **유연한 카메라 지원**: 웹캠, IP 카메라, 스마트폰 지원
- **환경변수 설정**: 코드 수정 없이 설정 변경 가능
- **로컬 모델 관리**: models/ 폴더에서 모델 통합 관리

## 시스템 아키텍처

```
카메라 입력
     |
     v
+------------------------------------------+
|           Python Application             |
|                                          |
|  VideoStream -> YOLODetector -> BoxTracker -> AnomalyDetector
|                                          |
|                                          v
|                                   UARTHandler
+------------------------------------------+
                     |
                     v
                   MCU -> 운전석 경고
```

## 프로젝트 구조

```
box_detection/
├── main.py                 # 메인 실행 파일
├── README.md
├── requirements.txt
│
├── config/                 # 설정
│   ├── __init__.py
│   └── settings.py         # 전역 설정 (환경변수 지원)
│
├── detection/              # 객체 감지
│   ├── __init__.py
│   ├── yolo_detector.py    # YOLO 모델 추론
│   └── box_info.py         # 박스 데이터 클래스
│
├── tracking/               # 추적 및 이상 감지
│   ├── __init__.py
│   ├── box_tracker.py      # IoU 기반 박스 추적
│   ├── movement_analyzer.py
│   └── anomaly_detector.py # 이상 감지 로직
│
├── communication/          # MCU 통신
│   ├── __init__.py
│   ├── uart_handler.py     # UART 시리얼 통신
│   └── message_formatter.py
│
├── camera/                 # 카메라 입력
│   ├── __init__.py
│   └── video_stream.py
│
├── utils/                  # 유틸리티
│   ├── __init__.py
│   ├── logger.py           # 로깅 (RotatingFileHandler)
│   └── visualization.py    # 시각화
│
├── models/                 # YOLO 모델 및 스크립트
│   ├── download_models.py  # 모델 다운로드 스크립트
│   ├── test_model.py       # 모델 테스트 스크립트
│   └── waste-classification.pt  # Waste 분류 모델 (기본)
│
└── logs/                   # 로그 파일 (자동 생성)
```

## 빠른 시작

### 1. 설치

```bash
# 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 모델 다운로드

```bash
python models/download_models.py
```

다운로드되는 모델:
- `waste-classification.pt` - 폐기물 분류 (cardboard 등 12클래스) - **기본 모델**
- `yolov8n.pt` - 일반 객체 검출 (옵션)
- `yolov8n-obb.pt` - 회전 바운딩 박스 (옵션)

### 3. 모델 테스트

```bash
# 웹캠으로 cardboard 검출 테스트
python models/test_model.py --webcam

# 이미지 파일로 테스트
python models/test_model.py --image test.jpg

# 모든 클래스 검출
python models/test_model.py --image test.jpg --all
```

### 4. 메인 시스템 실행

```bash
python main.py
```

조작법:
- `q`: 종료
- `r`: 추적기 초기화

## 환경변수 설정

코드 수정 없이 환경변수로 설정을 변경할 수 있습니다.

```bash
# 카메라 설정
export SMART_CARGO_CAMERA_SOURCE=0           # 웹캠 인덱스
export SMART_CARGO_CAMERA_SOURCE="http://192.168.0.10:8080/video"  # IP 카메라

# UART 설정
export SMART_CARGO_UART_PORT="/dev/cu.usbserial-1420"
export SMART_CARGO_UART_ENABLED=false        # 테스트 시 비활성화

# 검출 설정
export SMART_CARGO_CONFIDENCE_THRESHOLD=0.5
export SMART_CARGO_MOVEMENT_THRESHOLD=30.0   # 이동 감지 임계값 (픽셀)
export SMART_CARGO_ASPECT_RATIO_THRESHOLD=0.15  # 종횡비 변화 임계값

# 로깅
export SMART_CARGO_LOG_LEVEL=DEBUG
```

### 전체 환경변수 목록

| 환경변수 | 기본값 | 설명 |
|----------|--------|------|
| `SMART_CARGO_CAMERA_SOURCE` | IP URL | 카메라 소스 |
| `SMART_CARGO_FRAME_WIDTH` | 640 | 프레임 너비 |
| `SMART_CARGO_FRAME_HEIGHT` | 480 | 프레임 높이 |
| `SMART_CARGO_CONFIDENCE_THRESHOLD` | 0.5 | 검출 신뢰도 |
| `SMART_CARGO_MOVEMENT_THRESHOLD` | 30.0 | 이동 감지 (px) |
| `SMART_CARGO_ASPECT_RATIO_THRESHOLD` | 0.15 | 종횡비 변화 (15%) |
| `SMART_CARGO_UART_PORT` | /dev/cu.usbserial-1420 | UART 포트 |
| `SMART_CARGO_UART_BAUDRATE` | 9600 | 보드레이트 |
| `SMART_CARGO_UART_ENABLED` | true | UART 활성화 |
| `SMART_CARGO_LOG_LEVEL` | INFO | 로그 레벨 |
| `SMART_CARGO_ENABLE_VISUALIZATION` | true | 시각화 |

## 핵심 알고리즘

### 1. 위치 급변 감지
```
이동거리 = sqrt[(x2-x1)^2 + (y2-y1)^2]
if 이동거리 > 30px -> 이상 감지
```

### 2. 형태 변형 감지
```
종횡비 = width / height
변화율 = |현재 - 이전| / 이전
if 변화율 > 15% -> 충돌/쓰러짐 감지
```

### 3. 박스 매칭 (Hungarian Algorithm)
```
IoU = 교집합 / 합집합
Cost Matrix = 1.0 - IoU
최적 매칭: scipy.optimize.linear_sum_assignment
```

## UART 통신 프로토콜

### 메시지 포맷
```
<BOX_ID:ANOMALY_TYPE:MOVEMENT:ASPECT_RATIO_CHANGE>
```

### 이상 유형
| 유형 | 설명 |
|------|------|
| `NORMAL` | 정상 |
| `MOVEMENT` | 급격한 이동 |
| `COLLISION` | 형태 변형 (충돌/쓰러짐) |
| `BOTH` | 이동 + 변형 |

### 예시
```
<1:NORMAL:5.2:0.01>
<2:MOVEMENT:45.3:0.02>
<3:COLLISION:12.5:0.18>
```

### Arduino 파싱 예제
```cpp
void loop() {
    if (Serial.available()) {
        String msg = Serial.readStringUntil('\n');
        if (msg.indexOf("MOVEMENT") >= 0 || msg.indexOf("COLLISION") >= 0) {
            digitalWrite(WARNING_LED, HIGH);
            tone(BUZZER_PIN, 1000, 500);
        }
    }
}
```

## 모델 정보

### Waste Classification 모델
- **출처**: Hugging Face (kendrickfff/waste-classification-yolov8-ken)
- **클래스 (12개)**: battery, biological, brown-glass, **cardboard**, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass
- **cardboard 인덱스**: 3

### OBB 모델
- 회전된 바운딩 박스 검출
- 박스 기울기 각도 직접 측정 가능
- 쓰러짐 감지에 활용

## 성능

| 항목 | 수치 |
|------|------|
| 처리 속도 | 15-30 FPS (CPU) |
| 지연 시간 | < 100ms |
| 로그 관리 | 자동 로테이션 (10MB x 5개) |

## 문제 해결

### 카메라 연결 실패
```bash
# 웹캠 인덱스 변경
export SMART_CARGO_CAMERA_SOURCE=1
```

### UART 연결 실패
```bash
# 포트 권한 (Linux/macOS)
sudo chmod 666 /dev/ttyUSB0

# 테스트 모드 (UART 없이)
export SMART_CARGO_UART_ENABLED=false
```

### 모델 로드 실패
```bash
# 모델 재다운로드
python models/download_models.py
```

## 의존성

```
ultralytics>=8.0.0    # YOLOv8
opencv-python>=4.8.0  # 영상 처리
numpy>=1.24.0
pyserial>=3.5         # UART 통신
scipy>=1.10.0         # Hungarian Algorithm
```

## 라이선스

MIT License

## 기여자

- **개발자**: juhong
