"""
Waste Classification Model Test Script
Waste Classification 모델로 cardboard 클래스를 검출합니다.

사용법:
    python models/test_model.py --webcam
    python models/test_model.py --image test.jpg

사용 전 모델 다운로드:
    python models/download_models.py
"""

import argparse
import cv2
import os
import sys
import time
from pathlib import Path
from ultralytics import YOLO

# 이 스크립트가 models/ 폴더 안에 있으므로 상위 폴더가 PROJECT_ROOT
MODELS_DIR = Path(__file__).parent
PROJECT_ROOT = MODELS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    WASTE_MODEL_PATH,
    WASTE_MODEL_HF_ID,
    CONFIDENCE_THRESHOLD
)


# 클래스 정보 (waste-classification-yolov8-ken 모델)
CLASSES = {
    0: "battery",
    1: "biological",
    2: "brown-glass",
    3: "cardboard",
    4: "clothes",
    5: "green-glass",
    6: "metal",
    7: "paper",
    8: "plastic",
    9: "shoes",
    10: "trash",
    11: "white-glass"
}

# cardboard 클래스 인덱스
CARDBOARD_CLASS_ID = 3


def load_model():
    """models/ 폴더에서 모델 로드"""
    model_path = Path(WASTE_MODEL_PATH)

    # 로컬 모델 파일 확인
    if model_path.exists():
        print(f"Loading model from: {model_path}")
        try:
            model = YOLO(str(model_path))
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Failed to load local model: {e}")

    # 로컬 모델이 없으면 안내
    print(f"[ERROR] Model not found: {model_path}")
    print()
    print("Please download the model first:")
    print("    python download_models.py")
    print()
    print("Or download manually from Hugging Face:")
    print(f"    {WASTE_MODEL_HF_ID}")
    sys.exit(1)


def test_with_image(model, image_path: str, confidence: float = 0.5):
    """이미지 파일로 테스트"""
    print(f"\nTesting with image: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot read image from {image_path}")
        return

    # cardboard 클래스만 검출
    results = model(frame, conf=confidence, classes=[CARDBOARD_CLASS_ID], verbose=False)

    result = results[0]
    boxes = result.boxes

    print(f"Detected {len(boxes)} cardboard box(es)")

    # 바운딩 박스 그리기
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"cardboard: {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        print(f"  - Box at ({x1}, {y1}) to ({x2}, {y2}), confidence: {conf:.2f}")

    cv2.imshow("Cardboard Detection - Image", frame)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_with_webcam(model, camera_source: int = 0, confidence: float = 0.5):
    """웹캠으로 실시간 테스트"""
    print(f"\nStarting webcam test (source: {camera_source})")
    print("Press 'q' to quit")
    print(f"Detecting: cardboard (class {CARDBOARD_CLASS_ID})")
    print(f"Confidence threshold: {confidence}")

    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        # cardboard 클래스만 검출
        results = model(frame, conf=confidence, classes=[CARDBOARD_CLASS_ID], verbose=False)

        result = results[0]
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"cardboard: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # FPS 계산
        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            current_fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time = time.time()

        # 정보 패널
        cv2.rectangle(frame, (0, 0), (250, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Cardboard: {len(boxes)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Cardboard Detection - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam test ended")


def test_all_classes(model, image_path: str, confidence: float = 0.5):
    """모든 클래스 검출 (디버깅용)"""
    print(f"\nTesting ALL classes with image: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot read image from {image_path}")
        return

    results = model(frame, conf=confidence, verbose=False)

    result = results[0]
    boxes = result.boxes

    print(f"Detected {len(boxes)} object(s)")

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = CLASSES.get(cls_id, f"class_{cls_id}")
        color = colors[cls_id % len(colors)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name}: {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        print(f"  - {cls_name} at ({x1}, {y1}), confidence: {conf:.2f}")

    cv2.imshow("All Classes Detection", frame)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Test waste classification model (cardboard detection)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_model.py --webcam                    # 웹캠으로 테스트
  python test_model.py --webcam --source 1         # 두 번째 웹캠
  python test_model.py --image test.jpg            # 이미지 파일로 테스트
  python test_model.py --image test.jpg --all      # 모든 클래스 검출
  python test_model.py --webcam --conf 0.7         # 신뢰도 임계값 변경

Before running, download the model:
  python download_models.py
        """
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--webcam", action="store_true",
                              help="웹캠으로 실시간 테스트")
    source_group.add_argument("--image", type=str,
                              help="테스트할 이미지 파일 경로")

    parser.add_argument("--source", type=int, default=0,
                        help="웹캠 소스 인덱스 (기본값: 0)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"검출 신뢰도 임계값 (기본값: {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--all", action="store_true",
                        help="모든 클래스 검출")

    args = parser.parse_args()

    # 모델 로드
    model = load_model()

    # 테스트 실행
    if args.webcam:
        test_with_webcam(model, args.source, args.conf)
    elif args.image:
        if args.all:
            test_all_classes(model, args.image, args.conf)
        else:
            test_with_image(model, args.image, args.conf)


if __name__ == "__main__":
    main()
