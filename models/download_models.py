"""
Model Download Script
필요한 YOLO 모델들을 다운로드하여 models/ 폴더에 저장합니다.

사용법:
    python models/download_models.py
"""

import os
import sys
import shutil
from pathlib import Path

# 이 스크립트가 models/ 폴더 안에 있으므로 현재 폴더가 MODELS_DIR
MODELS_DIR = Path(__file__).parent
PROJECT_ROOT = MODELS_DIR.parent


def ensure_models_dir():
    """models 디렉토리 확인"""
    print(f"Models directory: {MODELS_DIR}")


def download_yolov8n():
    """YOLOv8n 기본 모델 다운로드"""
    from ultralytics import YOLO

    model_path = MODELS_DIR / "yolov8n.pt"

    if model_path.exists():
        print(f"[SKIP] yolov8n.pt already exists")
        return True

    print("[DOWNLOAD] yolov8n.pt ...")
    try:
        # ultralytics가 자동으로 다운로드
        model = YOLO("yolov8n.pt")

        # 현재 디렉토리에 다운로드된 파일을 models/로 이동
        downloaded = Path("yolov8n.pt")
        if downloaded.exists():
            shutil.move(str(downloaded), str(model_path))
            print(f"[OK] Saved to {model_path}")
        else:
            # 캐시에서 복사
            cache_path = Path.home() / ".cache" / "ultralytics" / "yolov8n.pt"
            if cache_path.exists():
                shutil.copy(str(cache_path), str(model_path))
                print(f"[OK] Copied from cache to {model_path}")
            else:
                print(f"[WARN] Model loaded but file not found for copying")

        return True
    except Exception as e:
        print(f"[ERROR] Failed to download yolov8n.pt: {e}")
        return False


def download_yolov8n_obb():
    """YOLOv8n OBB 모델 다운로드 (회전 바운딩 박스)"""
    from ultralytics import YOLO

    model_path = MODELS_DIR / "yolov8n-obb.pt"

    if model_path.exists():
        print(f"[SKIP] yolov8n-obb.pt already exists")
        return True

    print("[DOWNLOAD] yolov8n-obb.pt ...")
    try:
        model = YOLO("yolov8n-obb.pt")

        downloaded = Path("yolov8n-obb.pt")
        if downloaded.exists():
            shutil.move(str(downloaded), str(model_path))
            print(f"[OK] Saved to {model_path}")
        else:
            cache_path = Path.home() / ".cache" / "ultralytics" / "yolov8n-obb.pt"
            if cache_path.exists():
                shutil.copy(str(cache_path), str(model_path))
                print(f"[OK] Copied from cache to {model_path}")

        return True
    except Exception as e:
        print(f"[ERROR] Failed to download yolov8n-obb.pt: {e}")
        return False


def download_waste_classification():
    """Waste Classification 모델 다운로드 (Hugging Face)"""
    model_path = MODELS_DIR / "waste-classification.pt"
    hf_model_id = "kendrickfff/waste-classification-yolov8-ken"
    hf_filename = "yolov8n-waste-12cls-best.pt"

    if model_path.exists():
        print(f"[SKIP] waste-classification.pt already exists")
        return True

    print(f"[DOWNLOAD] {hf_model_id} ...")
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"[INFO] Downloading {hf_filename} from Hugging Face...")
        downloaded_path = hf_hub_download(
            repo_id=hf_model_id,
            filename=hf_filename
        )
        
        # 다운로드된 파일을 models 폴더로 복사
        import shutil
        shutil.copy(downloaded_path, str(model_path))
        print(f"[OK] Saved to {model_path}")
        
        return True
        
    except ImportError:
        print("[ERROR] huggingface_hub not installed")
        print("[INFO] Install it with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to download waste-classification model: {e}")
        return False


def main():
    print("=" * 60)
    print("Smart Cargo Care - Model Downloader")
    print("=" * 60)

    # models 디렉토리 생성
    ensure_models_dir()
    print()

    # 모델 다운로드
    results = {}

    print("[1/3] YOLOv8n (General Object Detection)")
    results["yolov8n"] = download_yolov8n()
    print()

    print("[2/3] YOLOv8n-OBB (Oriented Bounding Box)")
    results["yolov8n-obb"] = download_yolov8n_obb()
    print()

    print("[3/3] Waste Classification (Hugging Face)")
    results["waste"] = download_waste_classification()
    print()

    # 결과 요약
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)

    for model_name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {model_name}: {status}")

    print()
    print("Models directory:", MODELS_DIR)
    print()

    # models 폴더 내용 표시
    print("Downloaded models:")
    for f in MODELS_DIR.glob("*.pt"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
