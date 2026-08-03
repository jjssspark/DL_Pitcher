"""
LSTM 기반 다음 구종 예측 모델
투수/타자 Embedding + 상황 컨텍스트 + 투구 시퀀스 결합
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

NUM_PITCH_CLASSES = 8
PITCHER_VOCAB = 2000
BATTER_VOCAB = 3000
EMBED_DIM = 16


def build_model(seq_len: int, seq_feat_dim: int, ctx_dim: int) -> keras.Model:
    """
    모델 구조:
      투수 ID  ──→ Embedding ──┐
      타자 ID  ──→ Embedding ──┤
                               ├──→ Concat ──→ Dense ──→ Softmax
      투구 시퀀스 ──→ LSTM ────┤
                               │
      상황 컨텍스트 ────────────┘

    Args:
        seq_len: 입력 시퀀스 길이 (직전 몇 개의 투구를 볼 것인가).
        seq_feat_dim: 시퀀스 한 스텝(투구 1개)의 피처 차원.
        ctx_dim: 상황 컨텍스트 피처 차원.
    """
    seq_input     = keras.Input(shape=(seq_len, seq_feat_dim), name="seq_input")
    ctx_input     = keras.Input(shape=(ctx_dim,),              name="ctx_input")
    pitcher_input = keras.Input(shape=(1,),                    name="pitcher_input")
    batter_input  = keras.Input(shape=(1,),                    name="batter_input")

    pitcher_emb = layers.Embedding(PITCHER_VOCAB, EMBED_DIM, name="pitcher_emb")(pitcher_input)
    pitcher_emb = layers.Flatten()(pitcher_emb)

    batter_emb = layers.Embedding(BATTER_VOCAB, EMBED_DIM, name="batter_emb")(batter_input)
    batter_emb = layers.Flatten()(batter_emb)

    lstm_out = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False), name="bilstm"
    )(seq_input)
    lstm_out = layers.Dropout(0.3)(lstm_out)

    ctx_out = layers.Dense(32, activation="relu", name="ctx_dense")(ctx_input)

    combined = layers.Concatenate()([lstm_out, ctx_out, pitcher_emb, batter_emb])
    combined = layers.Dense(64, activation="relu")(combined)
    combined = layers.Dropout(0.3)(combined)

    output = layers.Dense(NUM_PITCH_CLASSES, activation="softmax", name="output")(combined)

    model = keras.Model(
        inputs=[seq_input, ctx_input, pitcher_input, batter_input],
        outputs=output,
        name="pitch_predictor"
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _encode_ids(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """원본 MLB ID → 0~vocab_size 연속 인덱스로 변환"""
    unique = np.unique(ids)
    mapping = {orig: idx % vocab_size for idx, orig in enumerate(unique)}
    return np.array([mapping[i] for i in ids])


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """
    클래스 불균형 보정 — sklearn의 'balanced' 방식으로 희귀 구종(FS 등)의 loss 가중치를 높인다.

    Day 2 실험 결과 (`train(..., use_class_weight=True)` vs 기본값 비교, 2025 시즌 전체,
    seq_len=5 기준):
      - 클래스 가중치 없음: 전체 정확도 46.6%, macro-F1 39.2%
      - 클래스 가중치 적용: 전체 정확도 37.4%, macro-F1 38.4%
    FS/CU/CH 같은 희귀 구종의 recall은 크게 올라가지만(FS 39%→80%), 그 대가로 가장 많이
    던지는 FF(포심)의 recall이 70.5%→15.9%로 무너지면서 전체 정확도와 macro-F1이 함께
    떨어졌다. 즉 이 데이터셋에서는 'balanced' 가중치가 단순한 트레이드오프가 아니라
    전반적인 손해였다. 그래서 `train()`의 기본값은 여전히 False이고, 이 옵션은 필요할 때
    켜서 실험해볼 수 있도록 남겨둔다. 자세한 내용은 docs/blog/day2.md 참고.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def train(
    X_seq: np.ndarray,
    X_ctx: np.ndarray,
    y: np.ndarray,
    pitcher_ids: np.ndarray,
    batter_ids: np.ndarray,
    save_path: str = "models/pitch_predictor.h5",
    use_class_weight: bool = False,
    epochs: int = 50,
    batch_size: int = 256,
) -> tuple[keras.Model, keras.callbacks.History]:
    """
    BiLSTM 구종 예측 모델을 학습한다.

    Args:
        X_seq, X_ctx, y, pitcher_ids, batter_ids: `feature_engineering.build_sequences()` 출력.
        save_path: 최적 checkpoint를 저장할 경로.
        use_class_weight: True면 `_compute_class_weights()`로 계산한 클래스 가중치를
            적용한다. 기본값 False — Day 2 실험에서 이 데이터셋 기준으로는 오히려
            전체 정확도/macro-F1이 나빠지는 것을 확인했다 (`_compute_class_weights` docstring 참고).
        epochs: 최대 epoch 수 (EarlyStopping으로 조기 종료될 수 있음).
        batch_size: 배치 크기.
    """
    from sklearn.model_selection import train_test_split

    pitcher_enc = _encode_ids(pitcher_ids, PITCHER_VOCAB)
    batter_enc  = _encode_ids(batter_ids,  BATTER_VOCAB)

    (X_p_tr, X_p_val, X_b_tr, X_b_val,
     X_seq_tr, X_seq_val,
     X_ctx_tr, X_ctx_val,
     y_tr, y_val) = train_test_split(
        pitcher_enc, batter_enc, X_seq, X_ctx, y,
        test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(
        seq_len=X_seq.shape[1],
        seq_feat_dim=X_seq.shape[2],
        ctx_dim=X_ctx.shape[1],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint(save_path, save_best_only=True),
    ]

    class_weight: Optional[dict[int, float]] = (
        _compute_class_weights(y_tr) if use_class_weight else None
    )

    history = model.fit(
        x={"seq_input": X_seq_tr, "ctx_input": X_ctx_tr,
           "pitcher_input": X_p_tr, "batter_input": X_b_tr},
        y=y_tr,
        validation_data=(
            {"seq_input": X_seq_val, "ctx_input": X_ctx_val,
             "pitcher_input": X_p_val, "batter_input": X_b_val},
            y_val
        ),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
    )
    return model, history


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(".")
    import pandas as pd

    from feature_engineering import build_sequences

    root        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scaler_path = os.path.join(root, "models", "scaler.pkl")
    model_path  = os.path.join(root, "models", "pitch_predictor.h5")

    df = pd.read_csv(os.path.join(root, "data", "raw", "statcast_2025_full.csv"))
    X_seq, X_ctx, y, pitchers, batters, scaler = build_sequences(
        df, scaler_path=scaler_path
    )
    print(f"데이터 크기: {X_seq.shape}")

    model, history = train(X_seq, X_ctx, y, pitchers, batters, save_path=model_path)
    print("학습 완료")
