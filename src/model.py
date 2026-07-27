"""
LSTM 기반 다음 구종 예측 모델
투수/타자 Embedding + 상황 컨텍스트 + 투구 시퀀스 결합
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_PITCH_CLASSES = 8
PITCHER_VOCAB     = 2000
BATTER_VOCAB      = 3000
EMBED_DIM         = 16


def build_model(seq_len: int, seq_feat_dim: int, ctx_dim: int) -> keras.Model:
    """
    모델 구조:
      투수 ID  ──→ Embedding ──┐
      타자 ID  ──→ Embedding ──┤
                               ├──→ Concat ──→ Dense ──→ Softmax
      투구 시퀀스 ──→ LSTM ────┤
                               │
      상황 컨텍스트 ────────────┘
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


def _compute_class_weights(y: np.ndarray) -> dict:
    """클래스 불균형 보정 — 희귀 구종(FS 등)을 더 크게 학습"""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def train(X_seq, X_ctx, y, pitcher_ids, batter_ids, save_path: str = "models/pitch_predictor.h5"):
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

    history = model.fit(
        x={"seq_input": X_seq_tr, "ctx_input": X_ctx_tr,
           "pitcher_input": X_p_tr, "batter_input": X_b_tr},
        y=y_tr,
        validation_data=(
            {"seq_input": X_seq_val, "ctx_input": X_ctx_val,
             "pitcher_input": X_p_val, "batter_input": X_b_val},
            y_val
        ),
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
    )
    return model, history


if __name__ == "__main__":
    import sys, os
    sys.path.append(".")
    from feature_engineering import build_sequences
    import pandas as pd

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
