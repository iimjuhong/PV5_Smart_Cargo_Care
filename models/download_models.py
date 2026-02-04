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
    from ultralytics import YOLO

    model_path = MODELS_DIR / "waste-classification.pt"
    hf_model_id = "kendrickfff/waste-classification-yolov8-ken"

    if model_path.exists():
        print(f"[SKIP] waste-classification.pt already exists")
        return True

    print(f"[DOWNLOAD] {hf_model_id} ...")
    try:
        # Hugging Face에서 모델 로드
        model = YOLO(hf_model_id)

        # 모델을 로컬에 저장
        # ultralytics 모델은 export 없이 직접 저장 가능
        # model.save()는 없으므로 다른 방법 사용

        # Hugging Face 캐시에서 모델 파일 찾기
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"

        # 모델 ID로 캐시 디렉토리 찾기
        found = False
        for cache_dir in hf_cache.glob("models--kendrickfff--waste-classification-yolov8-ken*"):
            # snapshots 폴더에서 .pt 파일 찾기
            for pt_file in cache_dir.rglob("*.pt"):
                shutil.copy(str(pt_file), str(model_path))
                print(f"[OK] Copied from HF cache to {model_path}")
                found = True
                break
            if found:
                break

        if not found:
            # 직접 다운로드 시도
            try:
                from huggingface_hub import hf_hub_download
                downloaded_path = hf_hub_download(
                    repo_id=hf_model_id,
                    filename="best.pt",  # 또는 model.pt
                    local_dir=str(MODELS_DIR),
                    local_dir_use_symlinks=False
                )
                # 파일명 변경
                if Path(downloaded_path).exists():
                    shutil.move(downloaded_path, str(model_path))
                    print(f"[OK] Downloaded to {model_path}")
                    found = True
            except Exception as hf_err:
                print(f"[WARN] huggingface_hub download failed: {hf_err}")

        if not found:
            # 마지막 방법: 모델 객체에서 저장
            print("[INFO] Saving model directly...")
            # 임시 추론 후 모델 가중치 저장
            import tempfile
            import numpy as np

            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = model(dummy_img, verbose=False)

            # 현재 디렉토리에 다운로드된 파일 확인
            for f in Path(".").glob("*.pt"):
                if "waste" in f.name.lower() or "best" in f.name.lower():
                    shutil.move(str(f), str(model_path))
                    print(f"[OK] Saved to {model_path}")
                    found = True
                    break

        if not found:
            print(f"[WARN] Model loaded but could not save to {model_path}")
            print(f"[INFO] You can manually copy the model file to {model_path}")

        return True
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
