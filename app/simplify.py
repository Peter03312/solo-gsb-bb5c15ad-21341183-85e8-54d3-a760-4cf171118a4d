"""无状态折线压缩：摆动门（斜率区间）算法 + 采样断档感知。

以每段首点为锚维护允许斜率区间 ``[low, high]``（初始 ``(-∞, +∞)``），逐点把
后续点当作候选端点：锚点到候选端点连线的斜率若落在区间内，等价于该连线使
此前所有省略点的数值误差都不超过 ``max_error``，候选点即可省略，并用其自身
约束收紧区间；首次超限时保留前一点作为新锚（新的压缩段），重置区间后继续
扫描。相邻采样间隔超过 ``max_gap_seconds`` 必须断段：断档两侧的点都保留，
且不在断档上连线。``max_error = 0`` 时斜率区间退化为单点，只有与锚点严格
共线的点才能被省略。

时间比较一律换算为 epoch 整数微秒，``max_gap_seconds`` 含等号语义
（间隔恰好等于阈值不断段）。
"""

from __future__ import annotations

from .models import parse_timestamp
from .resampling import epoch_microseconds, inclusive_gap_microseconds

_MICROSECONDS_PER_SECOND = 1_000_000


def simplify_series(
    timestamps: list[str],
    values: list[float],
    original_indices: list[int],
    *,
    max_error: float,
    max_gap_seconds: float,
    series_id: str | None = None,
) -> dict:
    """摆动门折线压缩。

    ``timestamps`` 必须已按时间排序且无重复（由接口层保证）；
    ``original_indices`` 为各排序后点在请求 ``points`` 数组中的原始索引。
    返回按时间升序的保留点（含原始索引与分段编号）、压缩前后点数、
    分段数，以及用输出折线回放全部原始点得到的最大绝对误差。
    """
    sample_us = [epoch_microseconds(parse_timestamp(ts)) for ts in timestamps]
    max_gap_us = inclusive_gap_microseconds(max_gap_seconds)
    n = len(values)

    # kept: (排序后索引, 分段编号)；不变式：kept[-1] 始终是当前锚点
    kept: list[tuple[int, int]] = [(0, 0)]
    segment = 0
    anchor = 0
    low = float("-inf")
    high = float("inf")

    j = 1
    while j < n:
        # 断档：相邻间隔超过 max_gap_seconds 必须断段，两侧点均保留
        if sample_us[j] - sample_us[j - 1] > max_gap_us:
            if kept[-1][0] != j - 1:
                kept.append((j - 1, segment))  # 断档前的点收尾当前段
            segment += 1
            kept.append((j, segment))  # 断档后的点开启新段
            anchor = j
            low, high = float("-inf"), float("inf")
            j += 1
            continue

        dt_seconds = (sample_us[j] - sample_us[anchor]) / _MICROSECONDS_PER_SECOND
        slope = (values[j] - values[anchor]) / dt_seconds
        if low <= slope <= high:
            # 连线 anchor→j 使此前所有省略点误差都不超限：j 暂时省略，
            # 并用 j 自身的约束收紧斜率区间（约束后续候选端点）
            half = max_error / dt_seconds
            if slope - half > low:
                low = slope - half
            if slope + half < high:
                high = slope + half
            j += 1
        else:
            # 首次超限：保留前一点作为新锚，重置区间后以新锚重新检查 j。
            # 区间非平凡说明锚点后已省略过点，故 j-1 > anchor 尚未保留。
            kept.append((j - 1, segment))
            anchor = j - 1
            low, high = float("-inf"), float("inf")

    if kept[-1][0] != n - 1:
        kept.append((n - 1, segment))

    # 用输出折线回放：同一分段内相邻保留点连线，逐原始点计算数值误差
    max_abs_error = 0.0
    for (a_idx, a_seg), (b_idx, b_seg) in zip(kept, kept[1:]):
        if a_seg != b_seg:
            continue  # 断档两侧不连线
        t_a = sample_us[a_idx]
        span = sample_us[b_idx] - t_a
        v_a = values[a_idx]
        dv = values[b_idx] - v_a
        for k in range(a_idx + 1, b_idx):
            recon = v_a + dv * ((sample_us[k] - t_a) / span)
            err = abs(values[k] - recon)
            if err > max_abs_error:
                max_abs_error = err

    points = [
        {
            "index": original_indices[idx],
            "timestamp": timestamps[idx],
            "value": values[idx],
            "segment_index": seg,
        }
        for idx, seg in kept
    ]

    return {
        "series_id": series_id,
        "max_error": max_error,
        "max_gap_seconds": max_gap_seconds,
        "input_point_count": n,
        "output_point_count": len(kept),
        "segment_count": segment + 1,
        "max_abs_error": max_abs_error,
        "points": points,
    }
