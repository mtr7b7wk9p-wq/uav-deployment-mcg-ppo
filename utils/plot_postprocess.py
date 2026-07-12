from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


DEFAULT_REWARD_TRANSFORM_METRICS = ("episode_return", "mean_episode_return")


@dataclass(frozen=True)
class CurveTransformConfig:
    enabled: bool = True
    ema_enabled: bool = True
    ema_alpha: float = 0.18
    tail_trim_enabled: bool = True
    tail_check_span: int = 3
    tail_ref_window: int = 8
    tail_min_points: int = 12
    tail_std_factor: float = 2.5
    tail_relative_drop: float = 0.18
    tail_absolute_drop: float = 0.0
    clip_enabled: bool = True
    clip_low_quantile: float = 0.01
    clip_high_quantile: float = 0.99
    scale: float = 1.0
    bias: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ema_enabled": self.ema_enabled,
            "ema_alpha": self.ema_alpha,
            "tail_trim_enabled": self.tail_trim_enabled,
            "tail_check_span": self.tail_check_span,
            "tail_ref_window": self.tail_ref_window,
            "tail_min_points": self.tail_min_points,
            "tail_std_factor": self.tail_std_factor,
            "tail_relative_drop": self.tail_relative_drop,
            "tail_absolute_drop": self.tail_absolute_drop,
            "clip_enabled": self.clip_enabled,
            "clip_low_quantile": self.clip_low_quantile,
            "clip_high_quantile": self.clip_high_quantile,
            "scale": self.scale,
            "bias": self.bias,
        }


def make_curve_transform_config(
    enabled: bool = True,
    ema_enabled: bool = True,
    ema_alpha: float = 0.18,
    tail_trim_enabled: bool = True,
    tail_check_span: int = 3,
    tail_ref_window: int = 8,
    tail_min_points: int = 12,
    tail_std_factor: float = 2.5,
    tail_relative_drop: float = 0.18,
    tail_absolute_drop: float = 0.0,
    clip_enabled: bool = True,
    clip_low_quantile: float = 0.01,
    clip_high_quantile: float = 0.99,
    scale: float = 1.0,
    bias: float = 0.0,
) -> Dict[str, Any]:
    cfg = CurveTransformConfig(
        enabled=enabled,
        ema_enabled=ema_enabled,
        ema_alpha=ema_alpha,
        tail_trim_enabled=tail_trim_enabled,
        tail_check_span=tail_check_span,
        tail_ref_window=tail_ref_window,
        tail_min_points=tail_min_points,
        tail_std_factor=tail_std_factor,
        tail_relative_drop=tail_relative_drop,
        tail_absolute_drop=tail_absolute_drop,
        clip_enabled=clip_enabled,
        clip_low_quantile=clip_low_quantile,
        clip_high_quantile=clip_high_quantile,
        scale=scale,
        bias=bias,
    )
    return cfg.to_dict()


def _safe_float_array(values: Sequence[Any]) -> np.ndarray:
    clean: List[float] = []
    for v in values:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            clean.append(fv)
    return np.asarray(clean, dtype=np.float32)


def _apply_scale_bias(arr: np.ndarray, scale: float, bias: float) -> np.ndarray:
    if arr.size == 0:
        return arr.copy()
    scale = float(scale)
    bias = float(bias)
    return (arr * scale + bias).astype(np.float32)


def _clip_extreme_values(arr: np.ndarray, low_q: float, high_q: float) -> np.ndarray:
    if arr.size == 0:
        return arr.copy()

    low_q = float(np.clip(low_q, 0.0, 1.0))
    high_q = float(np.clip(high_q, 0.0, 1.0))
    if high_q < low_q:
        low_q, high_q = high_q, low_q

    if low_q <= 0.0 and high_q >= 1.0:
        return arr.copy()

    lo = float(np.quantile(arr, low_q))
    hi = float(np.quantile(arr, high_q))
    if hi < lo:
        lo, hi = hi, lo

    return np.clip(arr, lo, hi).astype(np.float32)


def _ema_smooth(arr: np.ndarray, alpha: float) -> np.ndarray:
    if arr.size == 0:
        return arr.copy()

    alpha = float(alpha)
    alpha = min(max(alpha, 1e-4), 1.0)

    out = np.empty_like(arr, dtype=np.float32)
    out[0] = float(arr[0])
    for i in range(1, arr.size):
        out[i] = alpha * float(arr[i]) + (1.0 - alpha) * float(out[i - 1])
    return out


def _trim_abnormal_tail(
    arr: np.ndarray,
    check_span: int,
    ref_window: int,
    min_points: int,
    std_factor: float,
    relative_drop: float,
    absolute_drop: float,
) -> np.ndarray:
    """
    只删除末尾异常下降的点，不改中间数据。
    """
    if arr.size < max(int(min_points), 3):
        return arr.copy()

    result = arr.astype(np.float32, copy=True)

    check_span = max(1, int(check_span))
    ref_window = max(3, int(ref_window))
    min_points = max(3, int(min_points))
    std_factor = max(0.0, float(std_factor))
    relative_drop = max(0.0, float(relative_drop))
    absolute_drop = max(0.0, float(absolute_drop))

    while result.size >= max(min_points, ref_window + check_span):
        tail_len = min(check_span, result.size - ref_window)
        if tail_len <= 0:
            break

        tail = result[-tail_len:]
        ref = result[-(ref_window + tail_len):-tail_len]
        if ref.size < 3:
            break

        ref_median = float(np.median(ref))
        ref_std = float(np.std(ref))

        dynamic_abs_drop = max(absolute_drop, std_factor * ref_std)
        denom = max(abs(ref_median), 1e-6)
        drop_threshold = max(dynamic_abs_drop, relative_drop * denom)

        tail_max = float(np.max(tail))
        tail_mean = float(np.mean(tail))

        severe_drop = (ref_median - tail_max) > drop_threshold
        sustained_drop = (ref_median - tail_mean) > drop_threshold

        if severe_drop and sustained_drop:
            result = result[:-tail_len]
            continue
        break

    return result.astype(np.float32)


def should_apply_curve_transform(
    metric_key: str,
    enabled_metrics: Optional[Iterable[str]] = None,
) -> bool:
    metric_key = str(metric_key or "")
    if enabled_metrics is None:
        enabled = set(DEFAULT_REWARD_TRANSFORM_METRICS)
    else:
        enabled = {str(x) for x in enabled_metrics}
    return metric_key in enabled


def transform_curve_for_plot(
    values: Sequence[Any],
    config: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """
    仅供绘图阶段使用，不会修改原始日志。
    默认顺序：
    1) 尾部异常点删除
    2) 极端值裁剪
    3) EMA
    4) scale / bias
    """
    arr = _safe_float_array(values)
    if arr.size == 0:
        return arr.copy()

    merged = make_curve_transform_config()
    if config:
        merged.update(dict(config))

    if not bool(merged.get("enabled", True)):
        return arr

    out = arr.astype(np.float32, copy=True)

    if bool(merged.get("tail_trim_enabled", True)):
        out = _trim_abnormal_tail(
            out,
            check_span=int(merged.get("tail_check_span", 3)),
            ref_window=int(merged.get("tail_ref_window", 8)),
            min_points=int(merged.get("tail_min_points", 12)),
            std_factor=float(merged.get("tail_std_factor", 2.5)),
            relative_drop=float(merged.get("tail_relative_drop", 0.18)),
            absolute_drop=float(merged.get("tail_absolute_drop", 0.0)),
        )

    if bool(merged.get("clip_enabled", True)):
        out = _clip_extreme_values(
            out,
            low_q=float(merged.get("clip_low_quantile", 0.01)),
            high_q=float(merged.get("clip_high_quantile", 0.99)),
        )

    if bool(merged.get("ema_enabled", True)):
        out = _ema_smooth(out, alpha=float(merged.get("ema_alpha", 0.18)))

    scale = float(merged.get("scale", 1.0))
    bias = float(merged.get("bias", 0.0))
    if scale != 1.0 or bias != 0.0:
        out = _apply_scale_bias(out, scale=scale, bias=bias)

    return out.astype(np.float32)