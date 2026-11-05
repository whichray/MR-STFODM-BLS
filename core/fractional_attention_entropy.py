# -*- coding: utf-8 -*-
"""
Fractional Attention Entropy feature extraction for ECG / time-series signals.

This module is a Python implementation of the MATLAB functions supplied in
"分数阶注意熵":
    - FracAttenEn.m
    - MultiscaleFracAttenEn.m
    - CMFrAttEn.m
    - RCMFrAttEn.m
    - HFrAttEn.m
    - TSMFrAttE.m

The public entry point is ``extract_fractional_attention_features``.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.special import gamma, psi


EPS = 1e-12


@dataclass(frozen=True)
class FAEConfig:
    fo: float = 0.2
    scale: int = 4
    hierarchy_level: int = 3
    methods: Tuple[str, ...] = ("fae", "hfae", "msfae", "cmfae", "rcmfae", "tsfae")
    nan_policy: str = "median"  # "median" or "zero"


def _as_signal_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    return x


def peak_find(x: np.ndarray) -> np.ndarray:
    """寻找局部极大值并处理平台峰，尽量对齐 MATLAB PeakFind 的思路。"""
    x = _as_signal_1d(x)
    n = x.size
    if n < 3:
        return np.asarray([], dtype=np.int64)

    idx: List[int] = []
    i = 1
    while i < n - 1:
        if x[i - 1] < x[i] and x[i] > x[i + 1]:
            idx.append(i)
            i += 1
        elif x[i - 1] < x[i] and x[i] == x[i + 1]:
            k = 1
            while i + k < n and x[i] == x[i + k]:
                k += 1
            if i + k < n and x[i] > x[i + k]:
                idx.append(i + (k - 1) // 2)
            i += max(1, k)
        else:
            i += 1
    return np.asarray(idx, dtype=np.int64)


def _interval_pdf(intervals: np.ndarray, n: int) -> np.ndarray:
    intervals = np.asarray(intervals, dtype=np.int64).reshape(-1)
    if n < 4 or intervals.size == 0:
        return np.asarray([], dtype=np.float64)
    # 对齐 MATLAB 的 histcounts(T, 1:length(x)-1)：分箱边界为 1,2,...,N-1。
    edges = np.arange(1, n, dtype=np.float64)
    if edges.size < 2:
        return np.asarray([], dtype=np.float64)
    counts, _ = np.histogram(intervals, bins=edges)
    total = counts.sum()
    if total <= 0:
        return np.asarray([], dtype=np.float64)
    return counts.astype(np.float64) / float(total)


def _fractional_entropy_from_pdf(pdf: np.ndarray, fo: float) -> float:
    p = np.asarray(pdf, dtype=np.float64).reshape(-1)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return np.nan
    try:
        val = -np.sum(p * ((p ** (-fo) / gamma(fo + 1.0)) * (np.log(p) + psi(1.0) - psi(1.0 - fo))))
    except Exception:
        return np.nan
    return float(val) if np.isfinite(val) else np.nan


def frac_attention_entropy(x: np.ndarray, fo: float = 0.2, return_pdfs: bool = False):
    """计算单条一维信号的分数阶注意熵。"""
    x = _as_signal_1d(x)
    n = x.size
    if not (-1.0 < float(fo) < 1.0):
        raise ValueError("fo should be in the open interval (-1, 1).")
    if n < 8:
        out = (np.nan, [], [], [], []) if return_pdfs else np.nan
        return out

    xmax = peak_find(x)
    xmin = peak_find(-x)
    if xmax.size < 2 or xmin.size < 2:
        out = (np.nan, [], [], [], []) if return_pdfs else np.nan
        return out

    txx = np.diff(xmax)
    tnn = np.diff(xmin)
    mixed = np.sort(np.concatenate([xmax, xmin]))
    temp = np.diff(mixed)
    if temp.size < 2:
        out = (np.nan, [], [], [], []) if return_pdfs else np.nan
        return out

    if xmax[0] < xmin[0]:
        txn = temp[0::2]
        tnx = temp[1::2]
    else:
        txn = temp[1::2]
        tnx = temp[0::2]

    pnx = _interval_pdf(tnx, n)  # min -> max
    pnn = _interval_pdf(tnn, n)  # min -> min
    pxx = _interval_pdf(txx, n)  # max -> max
    pxn = _interval_pdf(txn, n)  # max -> min

    enx = _fractional_entropy_from_pdf(pnx, fo)
    enn = _fractional_entropy_from_pdf(pnn, fo)
    exx = _fractional_entropy_from_pdf(pxx, fo)
    exn = _fractional_entropy_from_pdf(pxn, fo)
    vals = np.asarray([enn, exx, exn, enx], dtype=np.float64)
    fae = float(np.mean(vals)) if np.all(np.isfinite(vals)) else np.nan
    if return_pdfs:
        return fae, pnx, pnn, pxx, pxn
    return fae


def _coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    x = _as_signal_1d(x)
    scale = int(scale)
    if scale <= 1:
        return x.copy()
    n_blocks = x.size // scale
    if n_blocks <= 0:
        return np.asarray([], dtype=np.float64)
    return x[: n_blocks * scale].reshape(n_blocks, scale).mean(axis=1)


def multiscale_frac_attention_entropy(x: np.ndarray, scale: int = 4, fo: float = 0.2) -> np.ndarray:
    return np.asarray([frac_attention_entropy(_coarse_grain(x, s), fo) for s in range(1, int(scale) + 1)], dtype=np.float64)


def composite_multiscale_frac_attention_entropy(x: np.ndarray, scale: int = 4, fo: float = 0.2) -> np.ndarray:
    x = _as_signal_1d(x)
    out = []
    for s in range(1, int(scale) + 1):
        vals = []
        for offset in range(s):
            xs = _coarse_grain(x[offset:], s)
            vals.append(frac_attention_entropy(xs, fo))
        vals = np.asarray(vals, dtype=np.float64)
        out.append(float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan)
    return np.asarray(out, dtype=np.float64)


def _pad_pdf(pdf: np.ndarray, width: int) -> np.ndarray:
    pdf = np.asarray(pdf, dtype=np.float64).reshape(-1)
    out = np.zeros(width, dtype=np.float64)
    out[: min(width, pdf.size)] = pdf[: min(width, pdf.size)]
    return out


def refined_composite_multiscale_frac_attention_entropy(x: np.ndarray, scale: int = 4, fo: float = 0.2) -> np.ndarray:
    x = _as_signal_1d(x)
    result = []
    for s in range(1, int(scale) + 1):
        pdf_groups = [[], [], [], []]  # pnx, pnn, pxx, pxn
        for offset in range(s):
            xs = _coarse_grain(x[offset:], s)
            _, pnx, pnn, pxx, pxn = frac_attention_entropy(xs, fo, return_pdfs=True)
            for group, pdf in zip(pdf_groups, [pnx, pnn, pxx, pxn]):
                group.append(np.asarray(pdf, dtype=np.float64))
        entropies = []
        for group in pdf_groups:
            max_len = max((p.size for p in group), default=0)
            if max_len <= 0:
                entropies.append(np.nan)
                continue
            mat = np.vstack([_pad_pdf(p, max_len) for p in group])
            entropies.append(_fractional_entropy_from_pdf(mat.mean(axis=0), fo))
        entropies = np.asarray(entropies, dtype=np.float64)
        result.append(float(np.mean(entropies)) if np.all(np.isfinite(entropies)) else np.nan)
    return np.asarray(result, dtype=np.float64)


def _hierarchy_series(x: np.ndarray, level: int) -> List[np.ndarray]:
    x = _as_signal_1d(x)
    if x.size < 2:
        return [x]
    n_pow = 2 ** int(np.floor(np.log2(x.size)))
    x = x[:n_pow]
    series = [x]
    cursor = 0
    for _ in range(1, int(level)):
        current_layer = series[cursor:]
        cursor = len(series)
        # 每一层只继续分解上一层结果；当序列过短时立即停止。
        for arr in current_layer:
            arr = np.asarray(arr, dtype=np.float64)
            if arr.size < 2:
                continue
            even = arr[0::2]
            odd = arr[1::2]
            m = min(even.size, odd.size)
            avg = (even[:m] + odd[:m]) / 2.0
            diff = (even[:m] - odd[:m]) / 2.0
            series.extend([avg, diff])
    return series[: max(1, 2 ** int(level) - 1)]


def hierarchical_frac_attention_entropy(x: np.ndarray, hierarchy_level: int = 3, fo: float = 0.2) -> np.ndarray:
    return np.asarray([frac_attention_entropy(s, fo) for s in _hierarchy_series(x, int(hierarchy_level))], dtype=np.float64)


def timeshift_multiscale_frac_attention_entropy(x: np.ndarray, scale: int = 4, fo: float = 0.2) -> np.ndarray:
    """在每个尺度 k 上使用 k 个相移子序列计算时移多尺度分数阶注意熵。"""
    x = _as_signal_1d(x)
    result = []
    for k in range(1, int(scale) + 1):
        vals = [frac_attention_entropy(x[offset::k], fo) for offset in range(k)]
        vals = np.asarray(vals, dtype=np.float64)
        result.append(float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan)
    return np.asarray(result, dtype=np.float64)


def fractional_attention_feature_names(
    methods: Sequence[str] = ("fae", "hfae", "msfae", "cmfae", "rcmfae", "tsfae"),
    scale: int = 4,
    hierarchy_level: int = 3,
    prefix: str = "FrAttE",
) -> List[str]:
    names: List[str] = []
    for method in [m.lower() for m in methods]:
        if method == "fae":
            names.append(f"{prefix}_single")
        elif method == "hfae":
            names.extend([f"{prefix}_hier_l{i+1}" for i in range(max(1, 2 ** int(hierarchy_level) - 1))])
        elif method == "msfae":
            names.extend([f"{prefix}_ms_s{s}" for s in range(1, int(scale) + 1)])
        elif method == "cmfae":
            names.extend([f"{prefix}_cms_s{s}" for s in range(1, int(scale) + 1)])
        elif method == "rcmfae":
            names.extend([f"{prefix}_rcms_s{s}" for s in range(1, int(scale) + 1)])
        elif method == "tsfae":
            names.extend([f"{prefix}_ts_s{s}" for s in range(1, int(scale) + 1)])
        else:
            raise ValueError(f"Unknown fractional attention entropy method: {method!r}")
    return names


def extract_fractional_attention_features(
    signals: np.ndarray,
    fo: float = 0.2,
    scale: int = 4,
    hierarchy_level: int = 3,
    methods: Sequence[str] = ("fae", "hfae", "msfae", "cmfae", "rcmfae", "tsfae"),
    nan_policy: str = "median",
    progress_every: int = 0,
    verbose: bool = False,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, List[str]]:
    """Extract FAE-family features for a 2D signal matrix [n_samples, signal_length].

    progress_every > 0 时每处理 N 个样本打印一次进度。
    eps 由 config.py 的 FRACTIONAL_ATTENTION_EPS 传入，用于防止除零和概率归一化失败。
    """
    global EPS
    EPS = max(float(eps), 1e-300)

    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    if signals.ndim != 2:
        raise ValueError(f"signals must be 2D [n_samples, signal_length], got {signals.shape}")

    methods = tuple(m.lower() for m in methods)
    rows: List[np.ndarray] = []
    n_total = int(signals.shape[0])
    progress_every = int(progress_every or 0)
    t0 = time.time()
    if verbose:
        print(
            f"[FAE] Start extraction: n_samples={n_total}, signal_length={signals.shape[1]}, "
            f"methods={list(methods)}, fo={fo}, scale={scale}, hierarchy_level={hierarchy_level}",
            flush=True,
        )
    for i, x in enumerate(signals, start=1):
        feats: List[float] = []
        for method in methods:
            if method == "fae":
                feats.append(frac_attention_entropy(x, fo))
            elif method == "hfae":
                feats.extend(hierarchical_frac_attention_entropy(x, hierarchy_level, fo).tolist())
            elif method == "msfae":
                feats.extend(multiscale_frac_attention_entropy(x, scale, fo).tolist())
            elif method == "cmfae":
                feats.extend(composite_multiscale_frac_attention_entropy(x, scale, fo).tolist())
            elif method == "rcmfae":
                feats.extend(refined_composite_multiscale_frac_attention_entropy(x, scale, fo).tolist())
            elif method == "tsfae":
                feats.extend(timeshift_multiscale_frac_attention_entropy(x, scale, fo).tolist())
            else:
                raise ValueError(f"Unknown fractional attention entropy method: {method!r}")
        rows.append(np.asarray(feats, dtype=np.float64))
        if verbose and progress_every > 0 and (i == 1 or i % progress_every == 0 or i == n_total):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            remain = (n_total - i) / rate if rate > 0 else 0.0
            print(
                f"[FAE] processed {i}/{n_total} samples "
                f"({100.0 * i / max(n_total, 1):.1f}%), "
                f"elapsed={elapsed/60:.1f} min, eta={remain/60:.1f} min",
                flush=True,
            )

    X_fae = np.vstack(rows).astype(np.float32)
    names = fractional_attention_feature_names(methods, scale, hierarchy_level)

    # BLS 与 sklearn StandardScaler 无法处理 NaN/Inf。中位数填补可在少量短片段或平坦片段失败时
    # 保留该特征；若整列均为无效值，则回退为 0。
    X_fae = np.asarray(X_fae, dtype=np.float32)
    bad = ~np.isfinite(X_fae)
    if np.any(bad):
        policy = str(nan_policy or "median").lower()
        if policy == "zero":
            X_fae = np.nan_to_num(X_fae, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        elif policy == "median":
            X_work = X_fae.astype(np.float64)
            for j in range(X_work.shape[1]):
                col = X_work[:, j]
                finite = np.isfinite(col)
                fill = float(np.median(col[finite])) if np.any(finite) else 0.0
                col[~finite] = fill
                X_work[:, j] = col
            X_fae = X_work.astype(np.float32)
        else:
            raise ValueError("nan_policy must be 'median' or 'zero'.")
    return X_fae, names
