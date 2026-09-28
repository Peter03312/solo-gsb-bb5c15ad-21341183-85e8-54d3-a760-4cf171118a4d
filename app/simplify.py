"""无状态折线压缩：以每段首点为锚、维护允许斜率区间的保误差顺序扫描。

算法（Reumann-Witkam 风格，按实际时间差判定）：

- 以当前段首点（锚点 ``a``）为基准，对每个已暂时省略的点 ``p`` 计算其允许
  锚→候选端点连线落入的斜率区间
  ``[(v_p − ε − v_a)/Δt_p, (v_p + ε − v_a)/Δt_p]``（``Δt`` 为正的整数微秒差），
  所有省略点区间的交集就是当前的**允许斜率区间** ``[lo, hi]``；
- 逐点把候选端点 ``c`` 的斜率 ``(v_c − v_a)/Δt_c`` 代入交集：落在交集内即可行，
  该点暂时省略，并把它自身的允许区间并入交集；首次不可行时**保留前一点**作为
  新锚点、从新锚点重新开始扫描（候选点本身在新段保留）；
- 相邻原始采样间隔超过 ``max_gap_seconds`` 时**必须断段**（断档判定优先），
  断档两侧的点均保留；
- ``ε = 0`` 时区间退化为精确共线判定（有理数交叉积比较），共线点仍可省略；
- 扫描结束后回放输出折线（段内相邻保留点之间按实际时间线性插值），
  计算全部原始点的最大绝对误差。

时间一律换算为 epoch 整数微秒参与比较；斜率边界以有理数（分子浮点 / 分母整数）
保存，交叉积比较避免除法误差。
"""

from __future__ import annotations

from .models import parse_timestamp
from .resampling import epoch_microseconds, inclusive_gap_microseconds

_MICROSECONDS_PER_SECOND = 1_000_000

# 可行性比较的相对容差：仅在 ε>0 时启用，吸收交叉积浮点舍入（不放宽真实偏差）；
# ε=0 时严格比较（共线即共线，非共线不省略）。
_FEASIBILITY_REL_SLACK = 1e-12

# 有理数以 (分子, 正整数分母) 表示；None 分别表示 ±∞
Fraction = tuple[float, int] | None


def _frac_max(a: Fraction, b: Fraction) -> Fraction:
    """两个有理数取较大者（None 表示 −∞）。"""
    if a is None:
        return b
    if b is None:
        return a
    (an, ad), (bn, bd) = a, b
    return a if an * bd >= bn * ad else b


def _frac_min(a: Fraction, b: Fraction) -> Fraction:
    """两个有理数取较小者（None 表示 +∞）。"""
    if a is None:
        return b
    if b is None:
        return a
    (an, ad), (bn, bd) = a, b
    return a if an * bd <= bn * ad else b


def _within(
    num: float,
    den: int,
    lo: Fraction,
    hi: Fraction,
    rel_slack: float,
) -> bool:
    """判断 num/den 是否落在闭区间 [lo, hi] 内（None 端点开界）。

    交叉积比较（两侧分母均为正），rel_slack 提供与数量级自适应的浮点舍入容差。
    """
    if lo is not None:
        ln, ld = lo
        lhs = num * ld  # num/den ≥ ln/ld  ⇔  num·ld ≥ ln·den
        rhs = ln * den
        tol = rel_slack * max(1.0, abs(lhs), abs(rhs))
        if lhs + tol < rhs:
            return False
    if hi is not None:
        hn, hd = hi
        lhs = num * hd  # num/den ≤ hn/hd  ⇔  num·hd ≤ hn·den
        rhs = hn * den
        tol = rel_slack * max(1.0, abs(lhs), abs(rhs))
        if lhs > rhs + tol:
            return False
    return True


def simplify_polyline(
    timestamps: list[str],
    values: list[float],
    *,
    tolerance: float,
    max_gap_seconds: float,
    series_id: str | None = None,
) -> dict:
    """对折线做保误差压缩。

    ``timestamps`` 必须已按时间排序且无重复（由接口层保证）。
    ``tolerance`` 为非负有限误差上限（值单位），``max_gap_seconds`` 为正数，
    相邻采样间隔严格大于该值（整数微秒比较）即断段。
    """
    n = len(values)
    sample_us = [epoch_microseconds(parse_timestamp(ts)) for ts in timestamps]
    max_gap_us = inclusive_gap_microseconds(max_gap_seconds)
    rel_slack = 0.0 if tolerance == 0.0 else _FEASIBILITY_REL_SLACK

    # 每段在原始（排序后）序列中的起点；断段（容差或断档）时追加新起点
    segment_starts: list[int] = [0]
    gap_boundaries: list[dict] = []

    # 保留点：(原始索引, 所属段编号)
    kept: list[tuple[int, int]] = [(0, 0)]

    anchor = 0
    lo: Fraction = None  # −∞
    hi: Fraction = None  # +∞
    segment_index = 0

    for i in range(1, n):
        dt_i = sample_us[i] - sample_us[i - 1]
        gap_break = dt_i > max_gap_us

        dt_anchor = sample_us[i] - sample_us[anchor]
        slope_num = values[i] - values[anchor]
        feasible = (not gap_break) and _within(
            slope_num, dt_anchor, lo, hi, rel_slack
        )

        if not feasible:
            # 保留前一点作为当前段终点（尚未保留时）
            if kept[-1][0] != i - 1:
                kept.append((i - 1, segment_index))

            if gap_break:
                gap_boundaries.append(
                    {
                        "before_index": i - 1,
                        "after_index": i,
                        "before_time": timestamps[i - 1],
                        "after_time": timestamps[i],
                        "gap_seconds": dt_i / _MICROSECONDS_PER_SECOND,
                    }
                )

            # 候选点作为新段首点（新锚）保留，斜率区间重置
            segment_index += 1
            segment_starts.append(i)
            anchor = i
            lo, hi = None, None
            kept.append((i, segment_index))
        else:
            # 该点暂时省略：其允许斜率区间并入交集
            point_lo = (values[i] - values[anchor] - tolerance, dt_anchor)
            point_hi = (values[i] - values[anchor] + tolerance, dt_anchor)
            lo = _frac_max(lo, point_lo)
            hi = _frac_min(hi, point_hi)

    # 末点始终保留（输出折线必须终于最后一个观测点）
    if kept[-1][0] != n - 1:
        kept.append((n - 1, segment_index))

    # 逐段汇总（段内原始索引连续：容差断开时首尾相邻，断档时首尾相邻但有时间缺口）
    kept_by_segment: dict[int, list[int]] = {}
    for orig_idx, seg in kept:
        kept_by_segment.setdefault(seg, []).append(orig_idx)

    segments_out: list[dict] = []
    for seg in range(len(segment_starts)):
        start = segment_starts[seg]
        end = segment_starts[seg + 1] - 1 if seg + 1 < len(segment_starts) else n - 1
        members = kept_by_segment[seg]
        segments_out.append(
            {
                "segment_index": seg,
                "start_index": members[0],
                "end_index": members[-1],
                "start_time": timestamps[members[0]],
                "end_time": timestamps[members[-1]],
                "input_point_count": end - start + 1,
                "kept_point_count": len(members),
            }
        )

    points_out: list[dict] = []
    for out_idx, (orig_idx, seg) in enumerate(kept):
        points_out.append(
            {
                "index": out_idx,
                "original_index": orig_idx,
                "timestamp": timestamps[orig_idx],
                "value": values[orig_idx],
                "segment_index": seg,
            }
        )

    max_error, max_error_index = _replay_max_error(
        sample_us, values, kept_by_segment
    )

    return {
        "series_id": series_id,
        "tolerance": tolerance,
        "max_gap_seconds": max_gap_seconds,
        "input_point_count": n,
        "kept_point_count": len(kept),
        "segment_count": len(segment_starts),
        "gap_boundary_count": len(gap_boundaries),
        "max_abs_error": max_error,
        "max_error_index": max_error_index,
        "max_error_timestamp": timestamps[max_error_index],
        "points": points_out,
        "segments": segments_out,
        "gap_boundaries": gap_boundaries,
    }


def _replay_max_error(
    sample_us: list[int],
    values: list[float],
    kept_by_segment: dict[int, list[int]],
) -> tuple[float, int]:
    """回放输出折线：段内相邻保留点之间按实际时间线性插值，遍历全部原始点
    求最大绝对误差及其原始索引（并列时取最早点）。"""
    max_error = 0.0
    max_error_index = 0

    for seg in sorted(kept_by_segment):
        members = kept_by_segment[seg]
        for j in range(len(members) - 1):
            a, b = members[j], members[j + 1]
            t_a, t_b = sample_us[a], sample_us[b]
            v_a, v_b = values[a], values[b]
            span = t_b - t_a
            for q in range(a, b + 1):
                if q == a:
                    replayed = v_a
                else:
                    replayed = v_a + (v_b - v_a) * (
                        (sample_us[q] - t_a) / span
                    )
                error = abs(values[q] - replayed)
                if error > max_error:
                    max_error = error
                    max_error_index = q

    return max_error, max_error_index
