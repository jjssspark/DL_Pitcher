"""
MLB Statcast 데이터 수집 모듈
pybaseball을 통해 투구 데이터를 수집하고 저장
"""

import pandas as pd
import pybaseball
import os

pybaseball.cache.enable()

# 수집할 컬럼 (모델 입력에 필요한 것만)
COLUMNS = [
    "game_date",
    "pitcher",           # 투수 ID
    "batter",            # 타자 ID
    "pitch_type",        # 구종 (FF, SL, CH, CU, SI, FC, FS)
    "release_speed",     # 구속 (mph)
    "release_spin_rate", # 회전수
    "pfx_x",             # 좌우 변화량
    "pfx_z",             # 상하 변화량
    "plate_x",           # 홈플레이트 좌우 코스
    "plate_z",           # 홈플레이트 상하 코스
    "balls",             # 볼 카운트
    "strikes",           # 스트라이크 카운트
    "outs_when_up",      # 아웃 카운트
    "inning",            # 이닝
    "inning_topbot",     # 초/말
    "on_1b",             # 1루 주자
    "on_2b",             # 2루 주자
    "on_3b",             # 3루 주자
    "bat_score",         # 타자팀 점수
    "fld_score",         # 수비팀 점수
    "stand",             # 타자 좌/우타
    "p_throws",          # 투수 좌/우투
    "home_team",
    "away_team",
    "at_bat_number",     # 타석 번호 (시퀀스 구성용)
    "pitch_number",      # 투구 번호 (타석 내)
]


def collect_statcast_data(start_date: str, end_date: str, save_path: str = None) -> pd.DataFrame:
    """
    Statcast 데이터 수집

    Args:
        start_date: 시작일 "YYYY-MM-DD"
        end_date:   종료일 "YYYY-MM-DD"
        save_path:  저장 경로 (None이면 저장 안 함)
    """
    print(f"[수집] {start_date} ~ {end_date}")
    df = pybaseball.statcast(start_dt=start_date, end_dt=end_date)

    available = [c for c in COLUMNS if c in df.columns]
    df = df[available].copy()

    df = df.dropna(subset=["pitch_type"])
    df = df[df["pitch_type"] != ""]

    print(f"[완료] {len(df):,}개 투구 수집")
    print(f"[구종 분포]\n{df['pitch_type'].value_counts()}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[저장] {save_path}")

    return df


def collect_full_season(year: int = 2023) -> pd.DataFrame:
    """월별로 나눠 시즌 전체 데이터 수집 (메모리 절약)"""
    months = [
        (f"{year}-04-01", f"{year}-04-30"),
        (f"{year}-05-01", f"{year}-05-31"),
        (f"{year}-06-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-07-31"),
        (f"{year}-08-01", f"{year}-08-31"),
        (f"{year}-09-01", f"{year}-09-30"),
        (f"{year}-10-01", f"{year}-10-05"),
    ]

    dfs = []
    for start, end in months:
        save_path = f"data/raw/statcast_{start[:7]}.csv"
        if os.path.exists(save_path):
            print(f"[캐시] {save_path}")
            dfs.append(pd.read_csv(save_path))
        else:
            df = collect_statcast_data(start, end, save_path)
            dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(f"data/raw/statcast_{year}_full.csv", index=False)
    print(f"\n[시즌 전체] {len(full_df):,}개 투구")
    return full_df


if __name__ == "__main__":
    import argparse, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="시즌 전체 수집 (월별 분할, 약 20~30분)")
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()

    if args.full:
        months = [
            (f"{args.year}-04-01", f"{args.year}-04-30"),
            (f"{args.year}-05-01", f"{args.year}-05-31"),
            (f"{args.year}-06-01", f"{args.year}-06-30"),
            (f"{args.year}-07-01", f"{args.year}-07-31"),
            (f"{args.year}-08-01", f"{args.year}-08-31"),
            (f"{args.year}-09-01", f"{args.year}-09-30"),
            (f"{args.year}-10-01", f"{args.year}-10-05"),
        ]
        dfs = []
        for start, end in months:
            path = os.path.join(root, "data", "raw", f"statcast_{start[:7]}.csv")
            if os.path.exists(path):
                print(f"[캐시] {path}")
                dfs.append(pd.read_csv(path))
            else:
                dfs.append(collect_statcast_data(start, end, path))

        full_df = pd.concat(dfs, ignore_index=True)
        out = os.path.join(root, "data", "raw", f"statcast_{args.year}_full.csv")
        full_df.to_csv(out, index=False)
        print(f"\n[완료] {len(full_df):,}개 → {out}")
    else:
        save_path = os.path.join(root, "data", "raw", "statcast_2023_04_sample.csv")
        df = collect_statcast_data("2023-04-01", "2023-04-14", save_path)
        print(df.head())
