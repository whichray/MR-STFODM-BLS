# -*- coding: utf-8 -*-
"""MIT-BIH 心律失常数据库原始目录加载器。

将本地 PhysioNet/WFDB 风格的 .hea、.dat、.atr 文件转换为主实验使用的 npz。
输出包含基础特征 X、二分类标签 y、记录分组 groups、特征名、原始心拍片段、
MIT-BIH 原始符号、AAMI N/S/V/F/Q 标签、R 峰位置及记录名称。"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from robust_io import safe_save_npz


DEFAULT_AAMI5_GROUPS = {
    "N": ["N", "L", "R", "e", "j", "0"],
    "S": ["A", "a", "J", "S", "1"],
    "V": ["V", "E", "2"],
    "F": ["F", "3"],
    "Q": ["/", "f", "Q", "?", "4"],
}


def _cfg(cfg, name: str, default):
    return getattr(cfg, name, default)


def _resolve_path(path_like, base_dir: Path | None = None) -> Path:
    p = Path(path_like)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    return p.resolve()


def _parse_records(records_cfg, raw_dir: Path) -> List[str]:
    """返回不带扩展名的记录编号列表。"""
    if records_cfg is None or str(records_cfg).strip().lower() == "auto":
        stems = []
        for hea in sorted(raw_dir.glob("*.hea")):
            stem = hea.stem
            if (raw_dir / f"{stem}.dat").exists() and (raw_dir / f"{stem}.atr").exists():
                stems.append(stem)
        return stems

    if isinstance(records_cfg, str):
        records = [x.strip() for x in records_cfg.replace(";", ",").split(",") if x.strip()]
    else:
        records = [str(x).strip() for x in records_cfg if str(x).strip()]
    return records


def _build_symbol_to_aami(groups: Dict[str, Sequence[str]]) -> Dict[str, str]:
    out = {}
    for group, symbols in groups.items():
        for s in symbols:
            out[str(s)] = str(group)
    return out


def _keep_symbols(cfg, symbol_to_aami: Dict[str, str]) -> set:
    keep_cfg = _cfg(cfg, "MITBIH_KEEP_SYMBOLS", "aami")
    if keep_cfg is None or str(keep_cfg).strip().lower() in {"aami", "auto", "mapped"}:
        return set(symbol_to_aami.keys())
    if isinstance(keep_cfg, str):
        return {x.strip() for x in keep_cfg.replace(";", ",").split(",") if x.strip()}
    return {str(x) for x in keep_cfg}


def _normalize_segment(seg: np.ndarray, mode: str, std_eps: float = 1e-8) -> np.ndarray:
    seg = np.asarray(seg, dtype=np.float32).reshape(-1)
    mode = str(mode or "zscore").lower()
    if mode in {"none", "raw", "off", "false"}:
        return seg.astype(np.float32)
    if mode in {"zscore", "standard", "standardize"}:
        mu = float(np.mean(seg))
        sd = float(np.std(seg))
        if not np.isfinite(sd) or sd < float(std_eps):
            return (seg - mu).astype(np.float32)
        return ((seg - mu) / sd).astype(np.float32)
    if mode in {"minmax", "0_1"}:
        lo = float(np.min(seg))
        hi = float(np.max(seg))
        if hi - lo < float(std_eps):
            return np.zeros_like(seg, dtype=np.float32)
        return ((seg - lo) / (hi - lo)).astype(np.float32)
    raise ValueError("MITBIH_SEGMENT_NORMALIZE must be one of: zscore, minmax, none.")


def _make_feature_names(window_size: int, channel: int) -> np.ndarray:
    half = window_size // 2
    names = []
    for i in range(window_size):
        rel = i - half
        names.append(f"ecg_ch{channel}_t{rel:+04d}")
    return np.asarray(names, dtype=object)


def _compute_rr_features(
    rpeaks: np.ndarray,
    record_names: np.ndarray,
    sampling_rate_hz: float,
    local_window: int,
    min_interval_sec: float = 1e-6,
    fallback_sec: float = 1.0,
) -> np.ndarray:
    """在不使用类别标签的前提下计算六个节律特征。

RR 间期在每条 MIT-BIH 记录内部独立计算，避免跨记录连接；记录首尾缺失的前后
间期使用该记录的稳健中位数回填。"""
    rpeaks = np.asarray(rpeaks, dtype=np.float64).reshape(-1)
    record_names = np.asarray(record_names).astype(str).reshape(-1)
    if rpeaks.shape[0] != record_names.shape[0]:
        raise ValueError("rpeaks and record_names must have the same length.")
    fs = float(sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("MITBIH_SAMPLING_RATE_HZ must be a positive finite value.")

    out = np.zeros((rpeaks.shape[0], 6), dtype=np.float32)
    local_window = max(1, int(local_window))
    for record in np.unique(record_names):
        idx = np.flatnonzero(record_names == record)
        if idx.size == 0:
            continue
        idx = idx[np.argsort(rpeaks[idx], kind="stable")]
        pos = rpeaks[idx]
        intervals = np.diff(pos) / fs
        min_interval_sec = max(float(min_interval_sec), 1e-12)
        fallback_sec = max(float(fallback_sec), min_interval_sec)
        valid = intervals[np.isfinite(intervals) & (intervals > min_interval_sec)]
        fallback = float(np.median(valid)) if valid.size else fallback_sec

        for j, original_idx in enumerate(idx):
            pre_rr = float(intervals[j - 1]) if j > 0 and intervals[j - 1] > min_interval_sec else fallback
            post_rr = float(intervals[j]) if j < intervals.size and intervals[j] > min_interval_sec else fallback

            lo = max(0, j - local_window)
            hi = min(intervals.size, j + local_window)
            neighbourhood = intervals[lo:hi]
            neighbourhood = neighbourhood[np.isfinite(neighbourhood) & (neighbourhood > min_interval_sec)]
            local_mean = float(np.mean(neighbourhood)) if neighbourhood.size else fallback
            local_mean = max(local_mean, min_interval_sec)
            pre_rr = max(pre_rr, min_interval_sec)
            post_rr = max(post_rr, min_interval_sec)

            out[original_idx] = np.asarray([
                pre_rr,
                post_rr,
                local_mean,
                pre_rr / local_mean,
                post_rr / local_mean,
                post_rr / pre_rr,
            ], dtype=np.float32)
    return out


def _rr_feature_names() -> np.ndarray:
    return np.asarray([
        "rr_pre_s",
        "rr_post_s",
        "rr_local_mean_s",
        "rr_pre_over_local",
        "rr_post_over_local",
        "rr_post_over_pre",
    ], dtype=object)


def build_mitbih_feature_npz(raw_dir, output_npz, cfg, base_dir: Path | None = None, force: bool | None = None) -> Path:
    """将 MIT-BIH 原始目录转换为实验所需的 npz。

raw_dir 为 WFDB 文件目录，output_npz 为输出路径，cfg 提供 MITBIH_* 配置，
base_dir 用于解析相对路径，force 可覆盖配置中的强制重建选项。"""
    raw_dir = _resolve_path(raw_dir, base_dir)
    output_npz = _resolve_path(output_npz, base_dir)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    if force is None:
        force = bool(_cfg(cfg, "MITBIH_FORCE_REBUILD_NPZ", False))
    if output_npz.exists() and not force:
        print(f"[MITBIH] Reusing existing preprocessed npz -> {output_npz}", flush=True)
        return output_npz

    if not raw_dir.exists():
        raise FileNotFoundError(f"MIT-BIH raw directory not found: {raw_dir}")

    try:
        import wfdb  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Reading MIT-BIH raw WFDB files requires the Python package 'wfdb'.\n"
            "Install it in your experiment environment with: pip install wfdb\n"
            f"Original import error: {exc}"
        ) from exc

    records = _parse_records(_cfg(cfg, "MITBIH_RECORDS", "auto"), raw_dir)
    verbose = bool(_cfg(cfg, "MITBIH_VERBOSE", True))
    if verbose:
        print(f"[MITBIH] Raw directory: {raw_dir}", flush=True)
        print(f"[MITBIH] Records found: {len(records)} -> {records[:10]}{' ...' if len(records) > 10 else ''}", flush=True)
    if not records:
        raise FileNotFoundError(
            f"No MIT-BIH records found in {raw_dir}. Expected matching *.hea, *.dat and *.atr files."
        )

    aami_groups = _cfg(cfg, "MULTICLASS_AAMI_GROUPS_BY_N", {}).get(5, DEFAULT_AAMI5_GROUPS)
    symbol_to_aami = _build_symbol_to_aami(aami_groups)
    keep_symbols = _keep_symbols(cfg, symbol_to_aami)

    channel = int(_cfg(cfg, "MITBIH_SIGNAL_CHANNEL", 0))
    window_size = int(_cfg(cfg, "MITBIH_WINDOW_SIZE", 256))
    min_window_size = int(_cfg(cfg, "MITBIH_MIN_WINDOW_SIZE", 32))
    if window_size < min_window_size:
        raise ValueError(f"MITBIH_WINDOW_SIZE is too small; use at least {min_window_size} samples.")
    pre = _cfg(cfg, "MITBIH_PRE_R_SAMPLES", None)
    post = _cfg(cfg, "MITBIH_POST_R_SAMPLES", None)
    if pre is None or post is None:
        pre = window_size // 2
        post = window_size - int(pre)
    pre = int(pre)
    post = int(post)
    if pre <= 0 or post <= 0:
        raise ValueError("MITBIH_PRE_R_SAMPLES and MITBIH_POST_R_SAMPLES must be positive.")
    window_size = pre + post

    normalize_mode = str(_cfg(cfg, "MITBIH_SEGMENT_NORMALIZE", "zscore"))
    segment_std_eps = float(_cfg(cfg, "MITBIH_SEGMENT_STD_EPS", 1e-8))
    min_rr_interval_sec = float(_cfg(cfg, "MITBIH_MIN_RR_INTERVAL_SEC", 1e-6))
    rr_fallback_sec = float(_cfg(cfg, "MITBIH_RR_FALLBACK_SEC", 1.0))
    max_beats_per_record = int(_cfg(cfg, "MITBIH_MAX_BEATS_PER_RECORD", 0) or 0)
    max_total_beats = int(_cfg(cfg, "MITBIH_MAX_TOTAL_BEATS", 0) or 0)
    seed = int(_cfg(cfg, "SEED", 42))
    rng = np.random.default_rng(seed)

    segments: List[np.ndarray] = []
    y_binary: List[int] = []
    groups: List[str] = []
    beat_labels: List[str] = []
    y_aami: List[str] = []
    rpeaks: List[int] = []
    record_names: List[str] = []
    skipped_by_reason = {
        "unmapped_or_not_kept": 0,
        "edge_window": 0,
        "bad_channel": 0,
        "read_error": 0,
    }
    per_record_counts = {}

    t_build0 = time.time()
    for rec_i, rec in enumerate(records, start=1):
        rec_t0 = time.time()
        rec_path = str(raw_dir / rec)
        if verbose:
            print(f"[MITBIH] ({rec_i}/{len(records)}) reading record {rec} ...", flush=True)
        try:
            sig, fields = wfdb.rdsamp(rec_path)
            ann = wfdb.rdann(rec_path, "atr")
        except Exception as exc:
            skipped_by_reason["read_error"] += 1
            if verbose:
                print(f"[MITBIH]   read_error for {rec}: {exc}", flush=True)
            continue

        sig = np.asarray(sig, dtype=np.float32)
        if sig.ndim != 2 or channel >= sig.shape[1]:
            skipped_by_reason["bad_channel"] += 1
            continue

        rec_indices = []
        for sample, sym in zip(ann.sample, ann.symbol):
            sym = str(sym)
            if sym not in keep_symbols or sym not in symbol_to_aami:
                skipped_by_reason["unmapped_or_not_kept"] += 1
                continue
            center = int(sample)
            start = center - pre
            end = center + post
            if start < 0 or end > sig.shape[0]:
                skipped_by_reason["edge_window"] += 1
                continue
            rec_indices.append((start, end, center, sym, symbol_to_aami[sym]))

        if max_beats_per_record > 0 and len(rec_indices) > max_beats_per_record:
            keep_idx = np.sort(rng.choice(len(rec_indices), size=max_beats_per_record, replace=False))
            rec_indices = [rec_indices[i] for i in keep_idx]

        per_record_counts[rec] = len(rec_indices)
        if verbose:
            print(
                f"[MITBIH]   extracted {len(rec_indices)} beats from {rec} "
                f"in {time.time() - rec_t0:.1f}s",
                flush=True,
            )
        for start, end, center, sym, aami in rec_indices:
            seg = _normalize_segment(sig[start:end, channel], normalize_mode)
            segments.append(seg)
            y_binary.append(0 if aami == "N" else 1)
            groups.append(str(rec))
            beat_labels.append(sym)
            y_aami.append(aami)
            rpeaks.append(int(center))
            record_names.append(str(rec))

    if not segments:
        raise RuntimeError(
            "No ECG beats were extracted from the MIT-BIH raw directory. "
            f"Skipped counts: {skipped_by_reason}. Check MITBIH_RECORDS / MITBIH_KEEP_SYMBOLS / window settings."
        )

    X_segments = np.vstack(segments).astype(np.float32)
    y_binary_arr = np.asarray(y_binary, dtype=np.int64)
    groups_arr = np.asarray(groups, dtype=object)
    beat_labels_arr = np.asarray(beat_labels, dtype=object)
    y_aami_arr = np.asarray(y_aami, dtype=object)
    rpeaks_arr = np.asarray(rpeaks, dtype=np.int64)
    record_names_arr = np.asarray(record_names, dtype=object)

    add_rr = bool(_cfg(cfg, "MITBIH_ADD_RR_FEATURES", True))
    rr_features_arr = _compute_rr_features(
        rpeaks_arr,
        record_names_arr,
        sampling_rate_hz=float(_cfg(cfg, "MITBIH_SAMPLING_RATE_HZ", 360.0)),
        local_window=int(_cfg(cfg, "MITBIH_RR_LOCAL_WINDOW", 10)),
        min_interval_sec=min_rr_interval_sec,
        fallback_sec=rr_fallback_sec,
    ) if add_rr else np.zeros((X_segments.shape[0], 0), dtype=np.float32)

    if max_total_beats > 0 and X_segments.shape[0] > max_total_beats:
        # 调试模式下按二分类标签近似分层抽样。
        selected = []
        for lab in np.unique(y_binary_arr):
            idx = np.where(y_binary_arr == lab)[0]
            take = max(1, int(round(max_total_beats * len(idx) / len(y_binary_arr))))
            take = min(take, len(idx))
            selected.append(rng.choice(idx, size=take, replace=False))
        idx = np.sort(np.concatenate(selected))[:max_total_beats]
        X_segments = X_segments[idx]
        y_binary_arr = y_binary_arr[idx]
        groups_arr = groups_arr[idx]
        beat_labels_arr = beat_labels_arr[idx]
        y_aami_arr = y_aami_arr[idx]
        rpeaks_arr = rpeaks_arr[idx]
        record_names_arr = record_names_arr[idx]
        rr_features_arr = rr_features_arr[idx]

    X_mode = str(_cfg(cfg, "MITBIH_X_MODE", "raw")).lower()
    if X_mode in {"raw", "waveform", "segments"}:
        waveform_names = _make_feature_names(X_segments.shape[1], channel)
        if add_rr:
            X = np.hstack([X_segments, rr_features_arr]).astype(np.float32)
            feature_names = np.concatenate([waveform_names, _rr_feature_names()])
        else:
            X = X_segments.copy()
            feature_names = waveform_names
    elif X_mode in {"zeros", "placeholder"}:
        # 主要用于仅运行 FAE 的场景，后续 load_feature_dataset 会替换 X。
        X = np.zeros((X_segments.shape[0], 1), dtype=np.float32)
        feature_names = np.asarray(["placeholder_raw_dataset_feature"], dtype=object)
    else:
        raise ValueError("MITBIH_X_MODE must be 'raw' or 'placeholder'.")

    meta = {
        "DatasetBuilder": "mitbih_raw_loader.build_mitbih_feature_npz",
        "RawDir": str(raw_dir),
        "Records": records,
        "SignalChannel": channel,
        "WindowSize": int(X_segments.shape[1]),
        "PreRSamples": pre,
        "PostRSamples": post,
        "SegmentNormalize": normalize_mode,
        "XMode": X_mode,
        "RRFeaturesEnabled": bool(add_rr),
        "RRFeatureNames": _rr_feature_names().astype(str).tolist() if add_rr else [],
        "RRLocalWindow": int(_cfg(cfg, "MITBIH_RR_LOCAL_WINDOW", 10)),
        "SamplingRateHz": float(_cfg(cfg, "MITBIH_SAMPLING_RATE_HZ", 360.0)),
        "NumExtractedBeats": int(X_segments.shape[0]),
        "SkippedByReason": skipped_by_reason,
        "PerRecordCounts": per_record_counts,
        "AAMI5Groups": aami_groups,
    }

    if verbose:
        print(
            f"[MITBIH] Saving npz: signals={X_segments.shape}, rr={rr_features_arr.shape}, X={X.shape}, "
            f"y_binary={np.bincount(y_binary_arr) if y_binary_arr.size else []}, "
            f"elapsed={(time.time() - t_build0)/60:.1f} min",
            flush=True,
        )

    safe_save_npz(
        output_npz,
        X=X.astype(np.float32),
        y=y_binary_arr,
        groups=groups_arr,
        feature_names=feature_names,
        signals=X_segments.astype(np.float32),
        rr_features=rr_features_arr.astype(np.float32),
        rr_feature_names=_rr_feature_names(),
        beat_labels=beat_labels_arr,
        symbols=beat_labels_arr,
        y_aami=y_aami_arr,
        y_multiclass=y_aami_arr,
        rpeaks=rpeaks_arr,
        record_names=record_names_arr,
        mitbih_meta=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object),
    )

    # 在判定缓存有效前逐个读取归档成员，以便尽早发现
    # 写入中断或归档不完整，而不是等到后续实验深处才报错。
    try:
        with np.load(output_npz, allow_pickle=True) as verify:
            for key in verify.files:
                _ = verify[key]
    except Exception as exc:
        try:
            output_npz.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(f"MIT-BIH npz verification failed: {output_npz}. Rebuild was aborted safely.") from exc

    if verbose:
        print(f"[MITBIH] Saved and verified -> {output_npz}", flush=True)
    return output_npz
