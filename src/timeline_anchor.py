"""
영상 시각 -> 투구 인덱스 매핑. 균등 분할 대신 실제 관측 지점(앵커)을 쓴다.

앱은 영상 8231초를 320투구로 균등 분할해 25.72초마다 한 칸씩 넘겼다. 그런데 실제
중계의 투구 간격은 광고·리플레이·타자 교체 때문에 12초에서 626초까지 널뛴다.
균등 가정으로는 구조적으로 어긋나고 화면에는 "타임라인이 공 한 개 느리다"로 보인다
(TS-007, TS-008).

방송 스코어버그의 투수 투구수(P:N)를 읽으면 "이 시각 = 그 투수의 N번째 투구"를 알 수
있다. Statcast 투구 목록에서 투수별 누적 카운트를 세면 절대 인덱스가 나온다. 그
지점들을 앵커로 잡고 사이를 선형 보간한다.

앵커가 없으면 균등 분할과 정확히 같은 값을 준다 — 붙였다고 기존 동작이 사라지지 않는다.
"""

Anchor = tuple[float, int]      # (영상 시각 초, 투구 인덱스)


def pitcher_pitch_counts(pitches: list[dict]) -> list[int]:
    """각 투구가 그 투수의 몇 번째 투구인지 센다. 스코어버그의 P:N과 같은 값이다."""
    seen: dict[int, int] = {}
    counts = []
    for pitch in pitches:
        pid = pitch.get("pitcher_id", 0)
        seen[pid] = seen.get(pid, 0) + 1
        counts.append(seen[pid])
    return counts


def resolve_anchors(
    counters: list[tuple[float, int | None]],
    pitches: list[dict],
    video_duration_sec: float,
) -> list[Anchor]:
    """
    (영상 시각, P:N) 관측을 (영상 시각, 투구 인덱스) 앵커로 바꾼다.

    인덱스는 그 시각에 **대기 중인** 투구다. P:N이 보이면 N개를 던졌으므로 다음
    차례는 N+1번째 = 인덱스 N이다. 앱이 이 인덱스를 그렇게 쓴다.

    P:N은 투수별 카운트라 경기 전체에서 유일하지 않다 — 투수가 바뀌면 1부터 다시
    센다. 그래서 같은 N을 갖는 투구가 여럿이다. 균등 가정 추정치에 가장 가까운
    후보를 고른다. 균등 가정은 국소적으로 틀려도 전역 평균은 맞으므로 "어느 투수
    구간인가"를 고르는 데는 충분하다.

    시각과 인덱스가 함께 증가하는 것만 남긴다. OCR 오독은 실측에서 전부 감소
    방향이었고(16 -> 3, 16 -> 2), 시간이 흐르는데 투구가 되돌아갈 수는 없다.

    남길 것을 앞에서부터 그리디로 고르면 안 된다. 오독 하나가 뒤따르는 정상 관측을
    줄줄이 막는다 — 5 -> 12(오독) -> 6, 7, 8 이면 6, 7, 8 이 전부 버려진다. 실측에서
    관측 282개 중 129개가 이렇게 탈락했다. 그래서 가장 긴 증가 사슬을 고른다
    (같은 입력에서 앵커 153개 -> 200개).

    길이가 같은 사슬이 여럿이면 앞선 관측을 쓴다. 12 -> 20 과 3 -> 20 이 둘 다
    길이 2일 때 12 쪽을 고르는 것이 오독의 방향(감소)과 맞는다.
    """
    if not pitches or video_duration_sec <= 0:
        return []

    counts = pitcher_pitch_counts(pitches)
    by_count: dict[int, list[int]] = {}
    for idx, count in enumerate(counts):
        by_count.setdefault(count, []).append(idx)

    n = len(pitches)
    candidates: list[Anchor] = []
    for t, counter in sorted(counters, key=lambda row: row[0]):
        if counter is None:
            continue
        matches = by_count.get(counter)
        if not matches:
            continue
        guess = (t / video_duration_sec) * n
        thrown = min(matches, key=lambda i: abs(i - guess))
        # 던진 공이 아니라 **다음에 던질 공**을 가리킨다. 앱은 c_idx를 대기 중인
        # 투구로 쓴다 — 볼카운트는 그 공을 던지기 전 상태이고, "방금 던진 구종"은
        # c_idx-1이며, 예측 대상도 c_idx다. 여기서 던진 공을 주면 화면 전체가
        # 한 구씩 밀린다(실측: 5구까지 던진 지점에서 앱은 4구째 SL 인플레이 아웃을
        # 방금 던진 구로 띄웠고 볼카운트도 0-0이었다. 영상은 5구째 FF 95mph, 1-0).
        candidates.append((t, min(thrown + 1, n - 1)))

    return _longest_increasing_chain(candidates)


def _longest_increasing_chain(candidates: list[Anchor]) -> list[Anchor]:
    """시각과 인덱스가 함께 증가하는 가장 긴 부분수열. 동점이면 앞선 것을 쓴다.

    O(n^2)이다. 실사용 입력이 관측 수백 개이고 경기당 한 번만 푸는 값이라
    (앱에서는 캐시된다) 이 정도면 충분하다. 이분탐색판은 동점 처리를 여기처럼
    직접 정할 수 없다.
    """
    size = len(candidates)
    if size == 0:
        return []

    length = [1] * size
    previous = [-1] * size
    for k in range(size):
        t_k, i_k = candidates[k]
        for j in range(k):
            t_j, i_j = candidates[j]
            if t_j < t_k and i_j < i_k and length[j] + 1 > length[k]:
                length[k] = length[j] + 1
                previous[k] = j

    end = max(range(size), key=lambda k: (length[k], -k))
    chain: list[Anchor] = []
    while end != -1:
        chain.append(candidates[end])
        end = previous[end]
    return chain[::-1]


def index_at_time(
    t: float,
    anchors: list[Anchor],
    video_duration_sec: float,
    n_pitches: int,
) -> int:
    """
    영상 시각 t에서의 투구 인덱스. 앵커 사이는 선형 보간, 밖은 균등 가정으로 외삽.
    """
    if n_pitches <= 0 or video_duration_sec <= 0:
        return 0

    def clamp(value: float) -> int:
        return max(0, min(int(value), n_pitches - 1))

    if not anchors:
        return clamp((t / video_duration_sec) * n_pitches)

    # 양 끝을 (0, 0)과 (영상 끝, 마지막 투구)로 막아 외삽도 보간으로 처리한다.
    points: list[Anchor] = [(0.0, 0), *anchors, (video_duration_sec, n_pitches - 1)]

    for (t0, i0), (t1, i1) in zip(points, points[1:]):
        if t <= t1:
            if t1 <= t0:
                return clamp(i1)
            ratio = (t - t0) / (t1 - t0)
            return clamp(i0 + ratio * (i1 - i0))
    return n_pitches - 1


def time_at_index(
    index: int,
    anchors: list[Anchor],
    video_duration_sec: float,
    n_pitches: int,
) -> float:
    """
    index_at_time의 역함수. 투구로 이동할 때 영상도 같이 옮기는 데 쓴다.

    두 방향이 같은 매핑을 써야 한다. 표시는 앵커 보간인데 이동은 균등 분할이면,
    버튼을 누른 순간 영상이 엉뚱한 데로 가고 다음 보고에서 인덱스가 도로 끌려간다.
    """
    if n_pitches <= 0 or video_duration_sec <= 0:
        return 0.0

    if not anchors:
        return max(0.0, min((index / n_pitches) * video_duration_sec, video_duration_sec))

    points: list[Anchor] = [(0.0, 0), *anchors, (video_duration_sec, n_pitches - 1)]

    for (t0, i0), (t1, i1) in zip(points, points[1:]):
        if index <= i1:
            if i1 <= i0:
                return t1
            ratio = (index - i0) / (i1 - i0)
            return max(0.0, min(t0 + ratio * (t1 - t0), video_duration_sec))
    return video_duration_sec
