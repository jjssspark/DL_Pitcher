"""
Phase 2: YOLOv8 야구공 감지 커스텀 모델 훈련
실행: venv/bin/python3 src/train_yolo.py --workspace <ws> --project <proj> --version <n>
Roboflow 무료 계정 필요: https://app.roboflow.com (API 키는 .env의 ROBOFLOW_API_KEY)
"""

import os
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data", "yolo")
MODEL_DIR = os.path.join(ROOT, "models")
RUNS_DIR  = os.path.join(ROOT, "runs")


def resolve_api_key(cli_key: str | None) -> str:
    """CLI 인자 > .env 순. 키를 셸 히스토리에 남기지 않으려면 .env를 쓴다."""
    if cli_key:
        return cli_key

    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit(
            "ROBOFLOW_API_KEY가 없습니다. .env에 넣거나 --api-key로 넘기세요."
        )
    return key


def download_dataset(
    api_key: str, workspace: str, project: str, version: int = 1,
    dest_dir: str = DATA_DIR,
) -> str:
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
    os.makedirs(dest_dir, exist_ok=True)
    r = requests.get(link, timeout=300)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(dest_dir)

    yaml_path = os.path.join(dest_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml 없음: {dest_dir}")

    print(f"[완료] {yaml_path}")
    return dest_dir


def train(
    data_yaml: str, epochs: int = 50, imgsz: int = 640,
    project_dir: str = MODEL_DIR, name: str = "baseball_detector",
):
    """
    YOLOv8n fine-tuning
    - base: yolov8n.pt (COCO 사전학습)
    - epochs: 50 (M1 기준 약 20~40분)

    Ultralytics는 name이 이미 있으면 name2, name3...으로 자동 증가시킨다.
    반환 경로를 하드코딩하면 실제 산출물과 어긋나므로 results.save_dir를 쓴다.
    """
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        name=name,
        project=project_dir,
        patience=10,           # early stopping
        save=True,
        device="mps",          # Apple Silicon GPU
    )

    best_path = os.path.join(str(results.save_dir), "weights", "best.pt")
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
    parser.add_argument("--api-key",   default=None, help="Roboflow API 키 (없으면 .env)")
    parser.add_argument("--workspace", required=True, help="Roboflow 워크스페이스 slug")
    parser.add_argument("--project",   required=True, help="Roboflow 프로젝트 slug")
    parser.add_argument("--version",   type=int, default=1, help="데이터셋 버전 (기본값 1)")
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--imgsz",     type=int, default=640)
    parser.add_argument("--data-dir",  default=DATA_DIR,
                        help="데이터셋 저장 위치 (기본 data/yolo)")
    parser.add_argument("--project-dir", default=MODEL_DIR,
                        help="학습 산출물 위치 (기본 models/)")
    parser.add_argument("--name",      default="baseball_detector",
                        help="학습 run 이름")
    parser.add_argument("--skip-download", action="store_true",
                        help="데이터셋이 이미 있으면 다운로드 생략")
    parser.add_argument("--download-only", action="store_true",
                        help="다운로드만 하고 학습은 하지 않는다")
    args = parser.parse_args()

    # 1. 데이터셋 다운로드
    data_yaml = os.path.join(args.data_dir, "data.yaml")
    if args.skip_download and os.path.exists(data_yaml):
        print(f"[스킵] 데이터셋 이미 존재: {data_yaml}")
    else:
        loc = download_dataset(
            resolve_api_key(args.api_key), args.workspace, args.project,
            args.version, dest_dir=args.data_dir,
        )
        data_yaml = os.path.join(loc, "data.yaml")

    if args.download_only:
        raise SystemExit(0)

    # 2. 훈련
    best_model = train(
        data_yaml, epochs=args.epochs, imgsz=args.imgsz,
        project_dir=args.project_dir, name=args.name,
    )

    # 3. 검증
    validate(best_model, data_yaml)
