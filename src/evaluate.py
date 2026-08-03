"""
모델 평가 스크립트
실행: python src/evaluate.py

주의: `feature_engineering.SEQUENCE_LENGTH`(현재 3)와 학습에 사용한 시퀀스 길이가
다르면 평가 결과가 왜곡된다. 평가 대상 모델을 바꿀 때는 그 모델을 학습시킨 seq_len과
동일한 값으로 맞춰야 한다.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # 서버/헤드리스 환경에서도 안전하게 저장만 하도록
import matplotlib.pyplot as plt

# 한글 라벨(구종/축 제목)이 깨지지 않도록 한글 지원 폰트를 우선 사용
plt.rcParams["font.family"] = ["NanumGothic", "AppleGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_sequences, PITCH_CLASSES
from model import _encode_ids, PITCHER_VOCAB, BATTER_VOCAB

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(ROOT, "data", "raw", "statcast_2025_full.csv")
MODEL_PATH   = os.path.join(ROOT, "models", "pitch_predictor.h5")
SCALER_PATH  = os.path.join(ROOT, "models", "scaler.pkl")
FIG_DIR      = os.path.join(ROOT, "notebooks", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LABEL_NAMES = {v: k for k, v in PITCH_CLASSES.items()}


def load_model_and_data() -> tuple[
    tf.keras.Model, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """저장된 모델·scaler를 불러와, 학습 때와 동일한 split으로 test set을 재구성한다."""
    print("[로드] 데이터 및 모델...")
    import joblib
    scaler = joblib.load(SCALER_PATH)   # 학습 때 저장한 scaler 재사용

    df = pd.read_csv(DATA_PATH)
    X_seq, X_ctx, y, pitchers, batters, _ = build_sequences(df, scaler=scaler)

    pitcher_enc = _encode_ids(pitchers, PITCHER_VOCAB)
    batter_enc  = _encode_ids(batters,  BATTER_VOCAB)

    # 학습 때와 동일한 split → test set만 추출
    (_, X_p_val, _, X_b_val,
     _, X_seq_val,
     _, X_ctx_val,
     _, y_val) = train_test_split(
        pitcher_enc, batter_enc, X_seq, X_ctx, y,
        test_size=0.2, random_state=42, stratify=y
    )

    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[완료] test 샘플: {len(y_val):,}개")
    return model, X_seq_val, X_ctx_val, X_p_val, X_b_val, y_val


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]
) -> None:
    """행 기준(%) 정규화한 혼동행렬 히트맵을 저장한다."""
    cm  = confusion_matrix(y_true, y_pred)
    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("구종 예측 혼동행렬 (%)")
    ax.set_xlabel("예측 구종")
    ax.set_ylabel("실제 구종")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "07_confusion_matrix.png")
    plt.savefig(path)
    print(f"[저장] {path}")
    plt.show()


def plot_accuracy_by_class(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]
) -> None:
    """구종별 recall(=해당 구종을 실제로 맞춘 비율) 막대그래프를 저장한다."""
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    accs   = [report[l]["recall"] * 100 for l in labels if l in report]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels[:len(accs)], accs,
                  color=sns.color_palette("Set2", len(accs)))
    ax.axhline(y=np.mean(accs), color="red", linestyle="--",
               label=f"평균 {np.mean(accs):.1f}%")
    ax.set_title("구종별 예측 정확도 (Recall %)")
    ax.set_xlabel("구종")
    ax.set_ylabel("정확도 (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "08_accuracy_by_class.png")
    plt.savefig(path)
    print(f"[저장] {path}")
    plt.show()


def plot_top3_prediction(probs: np.ndarray, y_true: np.ndarray, n_samples: int = 5) -> None:
    """샘플 N개에 대해 예측 확률 상위 3개 시각화"""
    label_names = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    indices     = np.random.choice(len(y_true), n_samples, replace=False)

    fig, axes = plt.subplots(1, n_samples, figsize=(14, 4))
    for ax, idx in zip(axes, indices):
        prob     = probs[idx]
        top3_idx = np.argsort(prob)[::-1][:3]
        top3_lbl = [label_names[i] for i in top3_idx]
        top3_val = [prob[i] * 100 for i in top3_idx]
        colors   = ["#2ecc71" if top3_lbl[j] == LABEL_NAMES[y_true[idx]]
                    else "#3498db" for j in range(3)]

        ax.bar(top3_lbl, top3_val, color=colors)
        ax.set_title(f"실제: {LABEL_NAMES[y_true[idx]]}", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_ylabel("확률 (%)")

    plt.suptitle("예측 확률 Top-3 (초록=정답)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "09_top3_prediction.png")
    plt.savefig(path)
    print(f"[저장] {path}")
    plt.show()


if __name__ == "__main__":
    model, X_seq_val, X_ctx_val, X_p_val, X_b_val, y_val = load_model_and_data()

    # model.predict()는 Apple Silicon + TF2.21에서 hang 이슈 있음
    # model() 직접 호출로 우회
    print("[예측] 모델 추론 중...")
    inputs = {
        "seq_input":     tf.constant(X_seq_val),
        "ctx_input":     tf.constant(X_ctx_val),
        "pitcher_input": tf.constant(X_p_val),
        "batter_input":  tf.constant(X_b_val),
    }
    probs  = model(inputs, training=False).numpy()
    y_pred = np.argmax(probs, axis=1)

    present = sorted(np.unique(np.concatenate([y_val, y_pred])))
    labels  = [LABEL_NAMES[i] for i in present]

    acc = np.mean(y_pred == y_val) * 100
    print(f"\n전체 정확도: {acc:.2f}%")
    print("\n[분류 리포트]")
    print(classification_report(y_val, y_pred, target_names=labels))

    plot_confusion_matrix(y_val, y_pred, labels)
    plot_accuracy_by_class(y_val, y_pred, labels)
    plot_top3_prediction(probs, y_val)
    print(f"\n[완료] 그래프 저장: {FIG_DIR}")
