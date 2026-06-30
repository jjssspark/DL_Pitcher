"""
Phase 2: YOLOv8 야구공 감지 커스텀 모델 훈련
실행: python src/train_yolo.py --api-key YOUR_KEY
Roboflow 무료 계정 필요: https://app.roboflow.com
"""

import os
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data", "yolo")
MODEL_DIR = os.path.join(ROOT, "models")


def download_dataset(api_key: str, workspace: str, project: str, version: int = 1) -> str:
    """
    Roboflow REST API로 YOLOv8 데이터셋 zip 직접 다운로드
    """
    import requests, zipfile, io

    # 다운로드 링크 조회
    meta_url = f"https://api.roboflow.com/{workspace}/{project}/{version}/yolov8?api_key={api_key}"
    meta = requests.get(meta_url, timeout=30).json()
    link = meta.get("export", {}).get("link")
    if not link:
        raise RuntimeError(f"다운로드 링크 없음. 응답: {meta}")

    # zip 다운로드 & 압축 해제
    print(f"[다운로드] {link}")
    os.makedirs(DATA_DIR, exist_ok=True)
    r = requests.get(link, timeout=120)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(DATA_DIR)

    yaml_path = os.path.join(DATA_DIR, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml 없음: {DATA_DIR}")

    print(f"[완료] {yaml_path}")
    return DATA_DIR


def train(data_yaml: str, epochs: int = 50, imgsz: int = 640):
    """
    YOLOv8n fine-tuning
    - base: yolov8n.pt (COCO 사전학습)
    - epochs: 50 (M1 기준 약 20~40분)
    """
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        name="baseball_detector",
        project=MODEL_DIR,
        patience=10,           # early stopping
        save=True,
        device="mps",          # Apple Silicon GPU
    )

    best_path = os.path.join(MODEL_DIR, "baseball_detector", "weights", "best.pt")
    print(f"\n[훈련 완료] 최적 모델: {best_path}")
    return best_path


def validate(model_path: str, data_yaml: str):
    """훈련된 모델 검증"""
    from ultralytics import YOLO

    model   = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    print(f"\n[검증 결과]")
    print(f"  mAP50    : {metrics.box.map50:.3f}")
    print(f"  mAP50-95 : {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall   : {metrics.box.mr:.3f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 야구공 감지 모델 훈련")
    parser.add_argument("--api-key",   required=True, help="Roboflow API 키")
    parser.add_argument("--workspace", required=True, help="Roboflow 워크스페이스 slug")
    parser.add_argument("--project",   required=True, help="Roboflow 프로젝트 slug")
    parser.add_argument("--version",   type=int, default=1, help="데이터셋 버전 (기본값 1)")
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--imgsz",     type=int, default=640)
    parser.add_argument("--skip-download", action="store_true",
                        help="데이터셋이 이미 있으면 다운로드 생략")
    args = parser.parse_args()

    # 1. 데이터셋 다운로드
    data_yaml = os.path.join(DATA_DIR, "data.yaml")
    if args.skip_download and os.path.exists(data_yaml):
        print(f"[스킵] 데이터셋 이미 존재: {data_yaml}")
    else:
        loc       = download_dataset(args.api_key, args.workspace, args.project, args.version)
        data_yaml = os.path.join(loc, "data.yaml")

    # 2. 훈련
    best_model = train(data_yaml, epochs=args.epochs, imgsz=args.imgsz)

    # 3. 검증
    validate(best_model, data_yaml)
