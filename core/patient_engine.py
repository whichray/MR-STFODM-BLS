# Extracted verbatim function/class bodies; see provenance.json.
from __future__ import annotations
import hashlib
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from types import SimpleNamespace
import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_fscore_support, recall_score)
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from bls_classifier import BroadLearningSystem
from fractional_attention_entropy import extract_fractional_attention_features
from robust_io import safe_save_npz, file_sha256
C = SimpleNamespace()
EPS = 1e-12
TIE_TOLERANCE = 1e-4

def configure(parameters):
    global C, EPS, TIE_TOLERANCE
    C = SimpleNamespace(**parameters)
    EPS = float(getattr(C, "NUMERIC_EPS", 1e-12))
    TIE_TOLERANCE = float(getattr(C, "TIE_TOLERANCE", EPS))

def _make_bls_model(random_state: int, feature_indices=None, input_dim: Optional[int] = None) -> BroadLearningSystem:
    """按照本实验统一的随机映射与归一化策略构建 BLS。"""
    return BroadLearningSystem(
        N1=C.BLS_N1,
        N2=C.BLS_N2,
        N3=C.BLS_N3,
        C=C.BLS_C,
        random_state=int(random_state),
        feature_indices=feature_indices,
        input_dim=input_dim,
        feature_mapping_mode=str(getattr(C, "BLS_FEATURE_MAPPING_MODE", "stable_global")),
        input_scale_mode=str(getattr(C, "BLS_INPUT_SCALE_MODE", "sqrt_selected")),
        activation_clip=float(getattr(C, "BLS_ACTIVATION_CLIP", 60.0)),
        sample_weight_eps=float(getattr(C, "BLS_SAMPLE_WEIGHT_EPS", 1e-12)),
        softmax_clip=float(getattr(C, "BLS_SOFTMAX_CLIP", 60.0)),
        softmax_eps=float(getattr(C, "BLS_SOFTMAX_EPS", 1e-12)),
    )


def _compute_bls_sample_weight(y: np.ndarray):
    """仅根据训练集生成温和的 BLS 类别样本权重。

验证集和测试集不做重采样或重加权。默认 sqrt_balanced 在提高稀有类别影响的同时，
避免完全逆频率权重常见的少数类精确率下降。"""
    mode = str(getattr(C, "BLS_CLASS_WEIGHT", "none") or "none").strip().lower()
    y = np.asarray(y)
    if mode in {"none", "off", "false", "0"}:
        return None
    labels, counts = np.unique(y, return_counts=True)
    if labels.size <= 1:
        return None
    n = float(y.size)
    k = float(labels.size)
    inverse = {lab: n / (k * float(cnt)) for lab, cnt in zip(labels, counts)}
    if mode == "balanced":
        power = 1.0
    elif mode in {"sqrt_balanced", "sqrt", "moderate"}:
        power = float(getattr(C, "BLS_CLASS_WEIGHT_POWER", 0.50))
    else:
        raise ValueError("BLS_CLASS_WEIGHT must be one of: none, balanced, sqrt_balanced.")

    weights = np.asarray([inverse[v] ** power for v in y], dtype=np.float64)
    weights /= max(float(np.mean(weights)), EPS)
    lo = float(getattr(C, "BLS_CLASS_WEIGHT_MIN", 0.25))
    hi = float(getattr(C, "BLS_CLASS_WEIGHT_MAX", 8.0))
    if lo <= 0 or hi < lo:
        raise ValueError("BLS_CLASS_WEIGHT_MIN/MAX must satisfy 0 < min <= max.")
    weights = np.clip(weights, lo, hi)
    weights /= max(float(np.mean(weights)), EPS)
    return weights


def _class_f1_values_from_metrics(metrics: dict, class_ids) -> list:
    """按配置的类别编号提取 F1，并遵循零支持样本处理策略。"""
    requested = {int(v) for v in (class_ids or [])}
    ignore_zero_support = bool(getattr(C, "METRIC_IGNORE_ZERO_SUPPORT_CLASSES", False))
    values = []
    for row in metrics.get("ClassMetrics", []) or []:
        try:
            label = int(row.get("Label"))
        except Exception:
            continue
        if label not in requested:
            continue
        if ignore_zero_support and int(row.get("Support", 0) or 0) <= 0:
            continue
        values.append(float(row.get("F1", 0.0)))
    return values


def _minority_f1_from_metrics(metrics: dict) -> float:
    """按 config 中指定的少数/瓶颈类别计算平均 F1。

    如果 METRIC_IGNORE_ZERO_SUPPORT_CLASSES=True，则 y_true 中 support=0 的类别不参与
    minority 平均，避免把“该 fold 没有这个类别”错误地计为 0 分。
    """
    values = _class_f1_values_from_metrics(metrics, getattr(C, "FITNESS_MINORITY_CLASS_IDS", []) or [])
    return float(np.mean(values)) if values else float(metrics.get("MacroF1", 0.0))


def _hard_class_f1_from_metrics(metrics: dict) -> float:
    """计算显式配置的困难类别平均 F1，例如 AAMI 的 F 类。"""
    values = _class_f1_values_from_metrics(metrics, getattr(C, "FITNESS_HARD_CLASS_IDS", []) or [])
    return float(np.mean(values)) if values else _minority_f1_from_metrics(metrics)


def _worst_minority_f1_from_metrics(metrics: dict) -> float:
    """返回少数类别中的最低 F1，避免优化时牺牲某一个稀有类别。"""
    values = _class_f1_values_from_metrics(metrics, getattr(C, "FITNESS_MINORITY_CLASS_IDS", []) or [])
    return float(np.min(values)) if values else _minority_f1_from_metrics(metrics)


def _fitness_from_metrics(metrics: dict, selected_ratio: float) -> float:
    """统一适应度函数。accuracy_weighted_f1 模式用于 Accuracy-first 实验。"""
    mode = str(getattr(C, "FITNESS_MODE", "macro_f1") or "macro_f1").lower()
    macro_f1 = float(metrics.get("MacroF1", 0.0))
    if mode == "macro_f1":
        utility = macro_f1
        metrics["MinorityF1"] = np.nan
    elif mode == "macro_f1_minority_f1":
        minority_f1 = _minority_f1_from_metrics(metrics)
        utility = (
            float(getattr(C, "FITNESS_W_MACRO_F1", 0.75)) * macro_f1
            + float(getattr(C, "FITNESS_W_MINORITY_F1", 0.20)) * minority_f1
            + float(getattr(C, "FITNESS_W_BAL_ACC", 0.05)) * float(metrics.get("BalancedAccuracy", 0.0))
        )
        metrics["MinorityF1"] = float(minority_f1)
        metrics["HardF1"] = np.nan
        metrics["WorstMinorityF1"] = np.nan
    elif mode == "macro_f1_hard_minority_f1":
        minority_f1 = _minority_f1_from_metrics(metrics)
        hard_f1 = _hard_class_f1_from_metrics(metrics)
        worst_minority_f1 = _worst_minority_f1_from_metrics(metrics)
        utility = (
            float(getattr(C, "FITNESS_W_MACRO_F1", 0.40)) * macro_f1
            + float(getattr(C, "FITNESS_W_MINORITY_F1", 0.15)) * minority_f1
            + float(getattr(C, "FITNESS_W_HARD_F1", 0.30)) * hard_f1
            + float(getattr(C, "FITNESS_W_WORST_MINORITY_F1", 0.10)) * worst_minority_f1
            + float(getattr(C, "FITNESS_W_BAL_ACC", 0.05)) * float(metrics.get("BalancedAccuracy", 0.0))
        )
        metrics["MinorityF1"] = float(minority_f1)
        metrics["HardF1"] = float(hard_f1)
        metrics["WorstMinorityF1"] = float(worst_minority_f1)
    elif mode == "accuracy_weighted_f1":
        minority_weight = float(getattr(C, "FITNESS_W_MINORITY_F1", 0.0))
        minority_f1 = _minority_f1_from_metrics(metrics) if minority_weight > 0 else np.nan
        utility = (
            float(getattr(C, "FITNESS_W_ACCURACY", 0.75)) * float(metrics.get("Accuracy", 0.0))
            + float(getattr(C, "FITNESS_W_WEIGHTED_F1", 0.20)) * float(metrics.get("WeightedF1", 0.0))
            + float(getattr(C, "FITNESS_W_MACRO_F1", 0.05)) * macro_f1
            + float(getattr(C, "FITNESS_W_BAL_ACC", 0.0)) * float(metrics.get("BalancedAccuracy", 0.0))
            + minority_weight * (0.0 if np.isnan(minority_f1) else float(minority_f1))
        )
        metrics["MinorityF1"] = float(minority_f1) if not np.isnan(minority_f1) else np.nan
    else:
        raise ValueError(
            "FITNESS_MODE must be one of: 'macro_f1', 'macro_f1_minority_f1', "
            "'accuracy_weighted_f1', 'macro_f1_hard_minority_f1'."
        )
    metrics["FitnessUtility"] = float(utility)
    return float(1.0 - utility + float(C.LAMBDA_PENALTY) * float(selected_ratio))


@dataclass(frozen=True)
class FoldData:
    """保存一次实验数据划分。

group_cv 使用 train_idx 与 val_idx；repeated random 8:1:1 还保存 test_idx，
从而使验证集仅参与特征选择，最终指标来自未参与选择的测试集。"""
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: Optional[np.ndarray] = None
    split_protocol: str = "group_cv"
    # nested patient CV 下保存全部内层 (train, validation) 全局索引。
    # train_idx / val_idx 继续保留一个参考折，供最终预测头和旧接口使用。
    inner_splits: tuple = ()


def stable_seed(base: int, *parts: Any) -> int:
    text = "|".join([str(base)] + [str(p) for p in parts])
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % (2**31 - 1)


def _safe_scalar_to_str(x) -> str:
    """将 NumPy/Python 标量标签转换为稳定的字符串键。"""
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)
    if isinstance(x, np.generic):
        x = x.item()
    return str(x)


def _as_1d_label_array(arr, name: str, n_samples: int):
    arr = np.asarray(arr)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.shape[0] != n_samples:
        raise ValueError(f"Label array {name!r} length {arr.shape[0]} does not match X samples {n_samples}.")
    return arr


def _build_group_map(groups: dict) -> dict:
    mapping = {}
    for new_label, raw_values in (groups or {}).items():
        for v in raw_values:
            mapping[_safe_scalar_to_str(v)] = str(new_label)
    return mapping


def _encode_labels_to_int(label_values, class_names):
    class_names = [str(c) for c in class_names]
    class_to_int = {c: i for i, c in enumerate(class_names)}
    encoded = np.asarray([class_to_int[str(v)] for v in label_values], dtype=int)
    return encoded, class_to_int


def _configured_split_protocol_raw() -> str:
    return str(getattr(C, "SPLIT_PROTOCOL", "nested_patient_group_cv") or "nested_patient_group_cv").strip().lower()


def _protocol_four_class_enabled() -> bool:
    protocol = _configured_split_protocol_raw()
    if protocol == "aami_ds1_ds2":
        return bool(getattr(C, "AAMI_FORCE_FOUR_CLASS", True))
    if protocol in {"nested_patient_group_cv", "nested_group_cv", "patient_group_cv", "patient_cv"}:
        return bool(getattr(C, "PATIENT_FORCE_FOUR_CLASS", True))
    return False


def _protocol_class_labels():
    if _configured_split_protocol_raw() == "aami_ds1_ds2":
        return [str(v) for v in getattr(C, "AAMI_CLASS_LABELS", ["N", "S", "V", "F"])]
    return [str(v) for v in getattr(C, "PATIENT_CLASS_LABELS", ["N", "S", "V", "F"])]


def _effective_multiclass_num_classes() -> int:
    # 26 号代码的两个协议都固定为五类；不再允许旧的 3/4 类兼容路径。
    if str(getattr(C, "CLASSIFICATION_MODE", "aami_nsvfq")).lower() not in {"navlr", "aami_nsvfq"}:
        raise ValueError("CLASSIFICATION_MODE must be 'navlr' or 'aami_nsvfq'.")
    return 5


def _make_multiclass_labels(raw_labels):
    """构造 0..K-1 多分类编码，并返回保留掩码和标签元信息。"""
    n_classes = _effective_multiclass_num_classes()
    mode_name = str(getattr(C, "CLASSIFICATION_MODE", "aami_nsvfq")).lower()
    if mode_name == "navlr":
        # 原始 MIT-BIH 符号不是 AAMI 合并标签，必须先显式过滤到 NAVLR。
        labels = [str(v) for v in getattr(C, "NAVLR_CLASS_LABELS", ["N", "A", "V", "L", "R"])]
        mapping = {label: label for label in labels}
        raw = np.asarray(raw_labels).reshape(-1)
        raw_str = np.asarray([_safe_scalar_to_str(v) for v in raw], dtype=object)
        keep = np.asarray([v in mapping for v in raw_str], dtype=bool)
        mapped = raw_str.copy()
        encoded, class_to_int = _encode_labels_to_int(mapped[keep], labels)
        counts = {name: int(np.sum(mapped[keep] == name)) for name in labels}
        if any(v == 0 for v in counts.values()):
            raise ValueError(f"NAVLR source is missing one or more classes: {counts}")
        return encoded, keep, {
            "TaskType": "multiclass",
            "NumClasses": 5,
            "ClassificationMode": "navlr",
            "LabelMode": "raw_symbol_explicit",
            "ClassNames": labels,
            "ClassToInt": class_to_int,
            "IntToClass": {int(v): k for k, v in class_to_int.items()},
            "ClassDistributionByName": counts,
            "DroppedSamples": int(raw_str.shape[0] - keep.sum()),
            "GroupingUsed": {label: [label] for label in labels},
        }

    mode = str(getattr(C, "MULTICLASS_LABEL_MODE", "explicit") or "explicit").lower()
    raw = np.asarray(raw_labels).reshape(-1)
    raw_str = np.asarray([_safe_scalar_to_str(v) for v in raw], dtype=object)

    keep = np.ones(raw_str.shape[0], dtype=bool)
    mapped = None
    class_names = None
    grouping_used = None

    if mode == "aami":
        all_groups = getattr(C, "MULTICLASS_AAMI_GROUPS_BY_N", {}) or {}
        groups = all_groups.get(n_classes)
        if not groups:
            raise ValueError(f"No AAMI grouping configured for {n_classes} classes.")
        mapping = _build_group_map(groups)
        mapped = np.asarray([mapping.get(v, None) for v in raw_str], dtype=object)
        keep = np.asarray([v is not None for v in mapped], dtype=bool)
        class_names = list(groups.keys())
        grouping_used = groups

    elif mode == "custom_groups":
        groups = getattr(C, "MULTICLASS_CUSTOM_GROUPS", {}) or {}
        if not groups:
            raise ValueError("MULTICLASS_LABEL_MODE='custom_groups' but MULTICLASS_CUSTOM_GROUPS is empty.")
        mapping = _build_group_map(groups)
        mapped = np.asarray([mapping.get(v, None) for v in raw_str], dtype=object)
        keep = np.asarray([v is not None for v in mapped], dtype=bool)
        class_names = list(groups.keys())
        grouping_used = groups

    elif mode == "explicit":
        labels = [str(v) for v in (
            _protocol_class_labels()
            if _protocol_four_class_enabled()
            else (getattr(C, "MULTICLASS_CLASS_LABELS", []) or [])
        )]
        if len(labels) != n_classes:
            raise ValueError("explicit mode requires len(MULTICLASS_CLASS_LABELS) == MULTICLASS_NUM_CLASSES.")
        label_set = set(labels)
        keep = np.asarray([v in label_set for v in raw_str], dtype=bool)
        mapped = raw_str.copy()
        class_names = labels
        grouping_used = {lab: [lab] for lab in labels}

    elif mode in {"auto", "top_n"}:
        unique, counts = np.unique(raw_str, return_counts=True)
        order = np.argsort(-counts)
        unique_sorted = unique[order]
        counts_sorted = counts[order]

        if mode == "auto" and unique_sorted.size == n_classes:
            selected = list(unique_sorted)
        else:
            if unique_sorted.size < n_classes:
                raise ValueError(
                    f"Requested {n_classes} classes but source labels only contain {unique_sorted.size} unique labels."
                )
            selected = list(unique_sorted[:n_classes])

        selected_set = set(map(str, selected))
        keep = np.asarray([v in selected_set for v in raw_str], dtype=bool)
        mapped = raw_str.copy()
        # 按类别频数降序固定顺序，使 top_n 自动选类更稳定。
        class_names = [str(v) for v in selected]
        grouping_used = {str(v): [str(v)] for v in selected}

    else:
        raise ValueError("26 号代码只允许 explicit AAMI N/S/V/F/Q 或 navlr 原始符号模式。")

    if not bool(getattr(C, "MULTICLASS_DROP_UNMAPPED", True)) and not np.all(keep):
        unmapped = sorted(set(raw_str[~keep].tolist()))
        raise ValueError(f"Unmapped labels found: {unmapped[:20]}")

    mapped_kept = np.asarray(mapped[keep], dtype=object)
    counts_by_class = {name: int(np.sum(mapped_kept == name)) for name in class_names}
    min_count = int(getattr(C, "MULTICLASS_MIN_CLASS_COUNT", 1) or 1)
    too_small = {k: v for k, v in counts_by_class.items() if v < min_count}
    if too_small:
        raise ValueError(f"Some classes have fewer than MULTICLASS_MIN_CLASS_COUNT={min_count}: {too_small}")

    if len([k for k, v in counts_by_class.items() if v > 0]) != n_classes:
        raise ValueError(
            f"After mapping/filtering, not all requested classes are present. Counts: {counts_by_class}"
        )

    encoded, class_to_int = _encode_labels_to_int(mapped_kept, class_names)
    label_info = {
        "TaskType": "multiclass",
        "NumClasses": int(n_classes),
        "LabelMode": mode,
        "ClassNames": list(class_names),
        "ClassToInt": class_to_int,
        "IntToClass": {int(v): k for k, v in class_to_int.items()},
        "ClassDistributionByName": counts_by_class,
        "DroppedSamples": int(raw_str.shape[0] - keep.sum()),
        "GroupingUsed": grouping_used,
    }
    return encoded, keep, label_info


def _make_binary_labels(y_raw):
    """保留历史二分类行为，同时记录标签元信息。"""
    y_arr = np.asarray(y_raw)
    try:
        y_encoded = y_arr.astype(int)
        labels, counts = np.unique(y_encoded, return_counts=True)
        if labels.size != 2:
            # 即使 y 中包含多于两类，binary 任务也不能静默保留多分类标签。
            # 明确报错，避免在错误标签口径下继续运行。
            raise ValueError
        class_names = [str(v) for v in labels.tolist()]
        label_info = {
            "TaskType": "binary",
            "NumClasses": 2,
            "LabelMode": "binary_y",
            "ClassNames": class_names,
            "ClassToInt": {str(v): int(v) for v in labels.tolist()},
            "IntToClass": {int(v): str(v) for v in labels.tolist()},
            "ClassDistributionByName": {str(k): int(v) for k, v in zip(labels, counts)},
            "DroppedSamples": 0,
        }
        return y_encoded, np.ones(y_encoded.shape[0], dtype=bool), label_info
    except Exception:
        raw_str = np.asarray([_safe_scalar_to_str(v) for v in y_arr.reshape(-1)], dtype=object)
        unique, counts = np.unique(raw_str, return_counts=True)
        if unique.size != 2:
            raise ValueError(f"CLASSIFICATION_TASK='binary' requires exactly 2 labels, got {unique.size}: {unique.tolist()[:20]}")
        # 按标签字符串排序，保证编码结果可复现。
        class_names = sorted([str(v) for v in unique.tolist()])
        encoded, class_to_int = _encode_labels_to_int(raw_str, class_names)
        label_info = {
            "TaskType": "binary",
            "NumClasses": 2,
            "LabelMode": "binary_encoded",
            "ClassNames": class_names,
            "ClassToInt": class_to_int,
            "IntToClass": {int(v): k for k, v in class_to_int.items()},
            "ClassDistributionByName": {str(k): int(v) for k, v in zip(unique, counts)},
            "DroppedSamples": 0,
        }
        return encoded, np.ones(encoded.shape[0], dtype=bool), label_info


def _normalize_signal_matrix(arr, n_samples: int, key: str) -> np.ndarray:
    """将 ECG 信号规范为 [样本数, 信号长度]。"""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.shape[0] != n_samples:
            raise ValueError(f"Signal key {key!r} is 1D with length {arr.shape[0]}, not n_samples={n_samples}.")
        raise ValueError(f"Signal key {key!r} is 1D per sample is ambiguous; expected 2D [samples, length].")
    if arr.ndim == 2:
        if arr.shape[0] == n_samples:
            return arr.astype(np.float32)
        if arr.shape[1] == n_samples:
            return arr.T.astype(np.float32)
    raise ValueError(
        f"Signal key {key!r} must be 2D [n_samples, signal_length] or [signal_length, n_samples], "
        f"got {arr.shape} while n_samples={n_samples}."
    )


def _choose_fractional_attention_signal_source(data, X: np.ndarray, n_samples: int):
    """选择用于分数阶注意熵提取的原始 ECG 信号矩阵。"""
    key_cfg = str(getattr(C, "FRACTIONAL_ATTENTION_SOURCE_KEY", "auto") or "auto")
    if key_cfg.lower() != "auto":
        if key_cfg not in data:
            raise KeyError(f"FRACTIONAL_ATTENTION_SOURCE_KEY={key_cfg!r} not found. Available npz keys: {list(data.keys())}")
        return key_cfg, _normalize_signal_matrix(data[key_cfg], n_samples, key_cfg)

    candidates = list(getattr(C, "FRACTIONAL_ATTENTION_SOURCE_KEY_CANDIDATES", []) or [])
    for key in candidates:
        if key in data:
            return key, _normalize_signal_matrix(data[key], n_samples, key)

    if bool(getattr(C, "FRACTIONAL_ATTENTION_USE_X_AS_SIGNAL", False)):
        return "X", _normalize_signal_matrix(X, n_samples, "X")

    raise KeyError(
        "USE_FRACTIONAL_ATTENTION_ENTROPY=True, but no raw ECG signal key was found. "
        f"Tried: {candidates}. Available npz keys: {list(data.keys())}. "
        "Please store raw ECG beat/segment matrix in the npz, set FRACTIONAL_ATTENTION_SOURCE_KEY, "
        "or set FRACTIONAL_ATTENTION_USE_X_AS_SIGNAL=True only if X itself is raw waveform data."
    )


def _fae_source_signature(feature_file: Path, signals: np.ndarray) -> dict:
    stat = feature_file.stat()
    # 使用有界摘要检测数据变化，避免对整个归档做完整哈希，
    # 也避免额外分配一份 MIT-BIH 规模的数组。
    preview = np.ascontiguousarray(np.asarray(signals[: min(512, len(signals))], dtype=np.float32))
    digest = hashlib.md5(preview.tobytes()).hexdigest()
    return {
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "signal_digest": digest,
    }


def _fractional_attention_cache_path(feature_file: Path, signal_key: str) -> Path:
    methods = "-".join(map(str, getattr(C, "FRACTIONAL_ATTENTION_METHODS", [])))
    tag = (
        f"{feature_file.stem}_{signal_key}_fo{getattr(C, 'FRACTIONAL_ATTENTION_ORDER', 0.2)}"
        f"_s{getattr(C, 'FRACTIONAL_ATTENTION_SCALE', 4)}"
        f"_h{getattr(C, 'FRACTIONAL_ATTENTION_HIERARCHY_LEVEL', 3)}_{methods}"
    )
    digest = hashlib.md5(tag.encode("utf-8")).hexdigest()[:12]
    cache_dir = feature_file.parent / "__fae_cache__"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"fae_features_{digest}.npz"


def _deterministic_derangement(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Return a deterministic random permutation with no fixed points when possible.

    A plain random permutation leaves about one row unchanged on average.  For a
    shuffled-feature negative control, even a few unchanged rows retain their true
    sample-level HFAE correspondence.  This repair preserves a valid permutation and
    the complete joint row distribution while guaranteeing zero fixed points for
    n_rows >= 2.
    """
    n_rows = int(n_rows)
    permutation = np.asarray(rng.permutation(n_rows), dtype=np.int64)
    if n_rows < 2:
        return permutation
    fixed = np.flatnonzero(permutation == np.arange(n_rows, dtype=np.int64))
    if fixed.size == 1:
        i = int(fixed[0])
        j = (i + 1) % n_rows
        permutation[i], permutation[j] = permutation[j], permutation[i]
    elif fixed.size > 1:
        permutation[fixed] = np.roll(permutation[fixed].copy(), 1)
    if np.any(permutation == np.arange(n_rows, dtype=np.int64)):
        raise RuntimeError("Failed to construct a fixed-point-free HFAE permutation.")
    return permutation


def _apply_fractional_attention_control(
    X_fae: np.ndarray,
    fae_names,
) -> Tuple[np.ndarray, List[str], dict]:
    """根据配置构造真实、置乱或噪声 HFAE 输入。

    shuffled 使用同一个样本排列同时重排全部 HFAE 列，因此完整保留 HFAE
    的联合分布和层级间相关结构，只破坏它与原始心拍、标签及 ECG/RR 特征的
    样本对应关系。noise 则按每列均值和标准差生成独立高斯噪声，并可裁剪到
    真实 HFAE 的观测范围，用于排除“仅增加 7 个维度”造成的偶然影响。

    该变换在数据划分之前执行，且使用固定 seed；因此所有方法、fold 和 run
    会看到完全相同的控制输入，保证配对消融可复现。
    """
    X_fae = np.asarray(X_fae, dtype=np.float32)
    names = list(map(str, fae_names))
    variant = str(
        getattr(C, "FRACTIONAL_ATTENTION_VARIANT", "real") or "real"
    ).strip().lower()
    seed = int(getattr(C, "FRACTIONAL_ATTENTION_CONTROL_SEED", 20260716))
    if variant == "real":
        return X_fae.copy(), names, {
            "Variant": "real",
            "Seed": seed,
            "Transform": "none",
        }

    rng = np.random.default_rng(seed)
    if variant == "shuffled":
        permutation = _deterministic_derangement(X_fae.shape[0], rng)
        transformed = X_fae[permutation].copy()
        out_names = [f"CTRL_SHUFFLED_{name}" for name in names]
        fixed_points = int(np.sum(permutation == np.arange(X_fae.shape[0], dtype=np.int64)))
        return transformed, out_names, {
            "Variant": "shuffled",
            "Seed": seed,
            "Transform": "single_common_row_derangement",
            "FixedPointCount": fixed_points,
            "PermutationSHA256": hashlib.sha256(
                np.asarray(permutation, dtype=np.int64).tobytes()
            ).hexdigest(),
        }

    if variant == "noise":
        means = np.nanmean(X_fae.astype(np.float64), axis=0)
        stds = np.nanstd(X_fae.astype(np.float64), axis=0, ddof=0)
        stds = np.where(np.isfinite(stds) & (stds > 0.0), stds, 0.0)
        transformed = rng.normal(
            loc=means,
            scale=np.where(stds > 0.0, stds, 1.0),
            size=X_fae.shape,
        )
        zero_std = stds <= 0.0
        if np.any(zero_std):
            transformed[:, zero_std] = means[zero_std]
        clip_enabled = bool(
            getattr(C, "FRACTIONAL_ATTENTION_NOISE_CLIP_TO_RANGE", True)
        )
        if clip_enabled:
            col_min = np.nanmin(X_fae.astype(np.float64), axis=0)
            col_max = np.nanmax(X_fae.astype(np.float64), axis=0)
            transformed = np.minimum(col_max, np.maximum(col_min, transformed))
        transformed = np.asarray(transformed, dtype=np.float32)
        out_names = [f"CTRL_NOISE_{name}" for name in names]
        return transformed, out_names, {
            "Variant": "noise",
            "Seed": seed,
            "Transform": "gaussian_matched_mean_std",
            "ClipToObservedRange": clip_enabled,
            "ReferenceMean": means.astype(float).tolist(),
            "ReferenceStd": stds.astype(float).tolist(),
        }

    raise ValueError(
        f"Unsupported FRACTIONAL_ATTENTION_VARIANT={variant!r}; "
        "expected 'real', 'shuffled' or 'noise'."
    )


def _augment_with_fractional_attention_features(
    feature_file: Path,
    data,
    X,
    feature_names,
    keep_mask,
    raw_n_samples: Optional[int] = None,
    X_signal_candidate: Optional[np.ndarray] = None,
):
    """在 X 后追加分数阶注意熵特征，或用其替换 X。

多分类标签构造可能已筛选 X，但 npz 中的原始信号通常仍保持原始样本顺序，
因此先按原始样本数校验信号，再应用 keep_mask。"""
    if not bool(getattr(C, "USE_FRACTIONAL_ATTENTION_ENTROPY", False)):
        return X, feature_names, None

    raw_n_samples = int(raw_n_samples or len(keep_mask) or X.shape[0])
    X_signal_candidate = X if X_signal_candidate is None else X_signal_candidate
    signal_key, signals_all = _choose_fractional_attention_signal_source(data, X_signal_candidate, raw_n_samples)
    print(f"[FAE] Signal source={signal_key!r}, raw_signal_shape={signals_all.shape}", flush=True)
    keep_mask_arr = np.asarray(keep_mask, dtype=bool)
    if keep_mask_arr.shape[0] != signals_all.shape[0]:
        raise ValueError(
            f"keep_mask length {keep_mask_arr.shape[0]} does not match raw signals {signals_all.shape[0]}. "
            "This usually means the npz arrays have inconsistent sample counts."
        )
    signals = signals_all[keep_mask_arr]
    print(f"[FAE] Kept signal shape after label filtering={signals.shape}", flush=True)

    cache_path = _fractional_attention_cache_path(feature_file, signal_key)
    use_cache = bool(getattr(C, "FRACTIONAL_ATTENTION_CACHE", True))
    signature = _fae_source_signature(feature_file, signals)
    X_fae = None
    fae_names = None
    if use_cache and cache_path.exists():
        print(f"[FAE] Cache found -> {cache_path}", flush=True)
        try:
            with np.load(cache_path, allow_pickle=True) as cached:
                cache_ok = (
                    int(cached["n_samples"]) == signals.shape[0]
                    and int(cached["signal_length"]) == signals.shape[1]
                    and int(cached["source_size"]) == signature["source_size"]
                    and int(cached["source_mtime_ns"]) == signature["source_mtime_ns"]
                    and str(cached["signal_digest"].item()) == signature["signal_digest"]
                )
                if cache_ok:
                    X_fae = np.asarray(cached["X_fae"], dtype=np.float32)
                    fae_names = np.asarray(cached["feature_names"], dtype=object).astype(str).tolist()
                    print(f"[FAE] Loaded validated cache: shape={X_fae.shape}", flush=True)
                else:
                    print("[FAE] Cache belongs to a different source/setting; recomputing.", flush=True)
        except Exception as exc:
            print(f"[FAE] Cache unreadable ({exc}); recomputing.", flush=True)

    if X_fae is None:
        print(f"[FAE] Computing fractional attention entropy features... cache will be saved to {cache_path}", flush=True)
        X_fae, fae_names = extract_fractional_attention_features(
            signals,
            fo=float(getattr(C, "FRACTIONAL_ATTENTION_ORDER", 0.2)),
            scale=int(getattr(C, "FRACTIONAL_ATTENTION_SCALE", 4)),
            hierarchy_level=int(getattr(C, "FRACTIONAL_ATTENTION_HIERARCHY_LEVEL", 3)),
            methods=list(getattr(C, "FRACTIONAL_ATTENTION_METHODS", ["fae", "hfae", "msfae", "cmfae", "rcmfae", "tsfae"])),
            nan_policy=str(getattr(C, "FRACTIONAL_ATTENTION_NAN_POLICY", "median")),
            progress_every=int(getattr(C, "FRACTIONAL_ATTENTION_PROGRESS_EVERY", 1000) or 0),
            verbose=bool(getattr(C, "FRACTIONAL_ATTENTION_VERBOSE", True)),
            eps=float(getattr(C, "FRACTIONAL_ATTENTION_EPS", 1e-12)),
        )
        print(f"[FAE] Computed features: shape={X_fae.shape}", flush=True)
        if use_cache:
            safe_save_npz(
                cache_path,
                X_fae=X_fae.astype(np.float32),
                feature_names=np.asarray(fae_names, dtype=object),
                n_samples=int(signals.shape[0]),
                signal_length=int(signals.shape[1]),
                signal_key=str(signal_key),
                source_size=np.asarray(signature["source_size"], dtype=np.int64),
                source_mtime_ns=np.asarray(signature["source_mtime_ns"], dtype=np.int64),
                signal_digest=np.asarray(signature["signal_digest"], dtype=object),
            )
            print(f"[FAE] Saved validated cache -> {cache_path}", flush=True)

    X_fae, fae_names, control_info = _apply_fractional_attention_control(
        X_fae,
        fae_names,
    )
    print(
        "[FAE] Ablation variant="
        f"{control_info.get('Variant', 'real')!r}, seed={control_info.get('Seed')}",
        flush=True,
    )

    append = bool(getattr(C, "FRACTIONAL_ATTENTION_APPEND_TO_X", True))
    if append:
        X_new = np.hstack([np.asarray(X, dtype=np.float32), np.asarray(X_fae, dtype=np.float32)]).astype(np.float32)
        names_new = np.concatenate([np.asarray(feature_names, dtype=str), np.asarray(fae_names, dtype=str)])
    else:
        X_new = np.asarray(X_fae, dtype=np.float32)
        names_new = np.asarray(fae_names, dtype=str)

    info = {
        "Enabled": True,
        "SourceSignalKey": str(signal_key),
        "AppendToX": bool(append),
        "NumAddedFeatures": int(X_fae.shape[1]),
        "FeatureNames": list(map(str, fae_names)),
        "Order": float(getattr(C, "FRACTIONAL_ATTENTION_ORDER", 0.2)),
        "Scale": int(getattr(C, "FRACTIONAL_ATTENTION_SCALE", 4)),
        "HierarchyLevel": int(getattr(C, "FRACTIONAL_ATTENTION_HIERARCHY_LEVEL", 3)),
        "Methods": list(map(str, getattr(C, "FRACTIONAL_ATTENTION_METHODS", []))),
        "Variant": str(control_info.get("Variant", "real")),
        "ControlSeed": int(control_info.get("Seed", 0)),
        "ControlTransform": str(control_info.get("Transform", "none")),
        "ControlMetadata": control_info,
        "CacheFile": str(cache_path) if use_cache else "",
    }
    return X_new, names_new, info


def load_feature_dataset(feature_file: Path):
    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_file}\n"
            "Please edit FEATURE_FILE in config.py or copy the npz file into this folder."
        )
    try:
        data = np.load(feature_file, allow_pickle=True)
        # npz 默认惰性读取；这里强制读取全部成员，使不完整归档
        # 在此处给出明确的重建提示，而不是在标签处理深处才失败。
        for _key in data.files:
            _ = data[_key]
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"无法完整读取 npz 文件：{feature_file}\n"
            "文件可能在上一次预处理时被中断或已经损坏。请删除该 npz 和 __fae_cache__，"
            "把 DATASET_MODE 改为 'build_npz'、REBUILD_NPZ=True 后重新构建。"
        ) from exc

    # 优先读取实验四采用的标准格式。
    if "X" in data and "y" in data:
        X = np.asarray(data["X"], dtype=np.float32)
        y_raw_base = _as_1d_label_array(data["y"], "y", X.shape[0])
        base_label_key = "y"
    # 兼容早期会议版本的数据格式。
    elif "X" in data and "Y" in data:
        X = np.asarray(data["X"], dtype=np.float32)
        y_raw_base = _as_1d_label_array(data["Y"], "Y", X.shape[0])
        base_label_key = "Y"
    else:
        raise KeyError("NPZ must contain X and y, or X and Y.")

    # 在多分类筛选前保留原始 X 的引用。
    # 仅当 FRACTIONAL_ATTENTION_USE_X_AS_SIGNAL=True 时需要该引用。
    X_all_for_signal = X
    raw_n_samples = int(X.shape[0])

    if "groups" in data:
        groups = np.asarray(data["groups"], dtype=str)
    else:
        # 该回退仅用于可行性调试；MIT-BIH 正式实验应提供 record_id
        # 分组信息，以避免记录级数据泄漏。
        groups = np.asarray([f"sample_{i}" for i in range(X.shape[0])], dtype=str)

    if "feature_names" in data:
        feature_names = np.asarray(data["feature_names"], dtype=object).astype(str)
    else:
        feature_names = np.asarray([f"f{i}" for i in range(X.shape[1])], dtype=str)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if X.shape[0] != y_raw_base.shape[0] or X.shape[0] != groups.shape[0]:
        raise ValueError("X, y and groups must have the same number of samples.")
    if len(feature_names) != X.shape[1]:
        raise ValueError("feature_names length must match X.shape[1].")

    task_type = str(getattr(C, "CLASSIFICATION_TASK", "binary") or "binary").lower()
    if task_type == "multiclass":
        classification_mode = str(getattr(C, "CLASSIFICATION_MODE", "aami_nsvfq") or "aami_nsvfq").lower()
        if classification_mode == "navlr":
            label_source = str(getattr(C, "NAVLR_LABEL_KEY", "beat_labels"))
            if label_source not in data:
                raise KeyError(f"NAVLR requires {label_source!r} in the NPZ file.")
            raw_labels = _as_1d_label_array(data[label_source], label_source, X.shape[0])
        elif classification_mode == "aami_nsvfq":
            label_source = "y_aami"
            if label_source not in data:
                raise KeyError("AAMI N/S/V/F/Q mode requires y_aami in the NPZ file.")
            raw_labels = _as_1d_label_array(data[label_source], label_source, X.shape[0])
        else:
            raise ValueError("CLASSIFICATION_MODE must be 'navlr' or 'aami_nsvfq'.")
        y, keep_mask, label_info = _make_multiclass_labels(raw_labels)
        label_info["ClassificationMode"] = classification_mode
        label_info["SourceLabelKey"] = label_source
        label_info["BaseLabelKey"] = base_label_key
        if not np.all(keep_mask):
            X = X[keep_mask]
            groups = groups[keep_mask]
        # y 已由 _make_multiclass_labels 完成筛选与编码。
    elif task_type == "binary":
        y, keep_mask, label_info = _make_binary_labels(y_raw_base)
        label_info["SourceLabelKey"] = base_label_key
        label_info["BaseLabelKey"] = base_label_key
    else:
        raise ValueError("CLASSIFICATION_TASK must be either 'binary' or 'multiclass'.")

    X, feature_names, fae_info = _augment_with_fractional_attention_features(
        feature_file=feature_file,
        data=data,
        X=X,
        feature_names=feature_names,
        keep_mask=keep_mask,
        raw_n_samples=raw_n_samples,
        X_signal_candidate=X_all_for_signal,
    )
    if fae_info is not None:
        label_info["FractionalAttentionEntropy"] = fae_info

    if X.shape[0] != y.shape[0] or X.shape[0] != groups.shape[0]:
        raise ValueError("After label construction, X, y and groups have inconsistent sample counts.")
    if len(feature_names) != X.shape[1]:
        raise ValueError("After fractional attention entropy augmentation, feature_names length must match X.shape[1].")

    if _protocol_four_class_enabled():
        label_info["ProtocolClassOverride"] = {
            "Protocol": _split_protocol_name(),
            "ClassNames": _protocol_class_labels(),
            "Reason": "patient-independent evaluation excludes the paced-record-dominated Q class",
        }
    return X.astype(np.float32), y.astype(int), groups, feature_names, label_info


def apply_protocol_dataset_filter(X, y, record_groups, feature_names, label_info):
    """按固定协议排除不属于评价总体的记录；默认嵌套协议不做样本过滤。"""
    protocol = _split_protocol_name()
    if protocol == "nested_patient_group_cv" and bool(getattr(C, "PATIENT_EXCLUDE_PACED_RECORDS", True)):
        record_groups = np.asarray(record_groups).astype(str)
        excluded = {str(v) for v in getattr(C, "PATIENT_EXCLUDED_RECORDS", [])}
        keep = ~np.isin(record_groups, sorted(excluded))
        filtered_info = dict(label_info)
        filtered_info["ProtocolDatasetFilter"] = {
            "Protocol": protocol,
            "InputSamples": int(len(y)),
            "OutputSamples": int(np.sum(keep)),
            "ExcludedRecords": sorted(excluded),
        }
        return X[keep], y[keep], record_groups[keep], feature_names, filtered_info
    if protocol != "aami_ds1_ds2":
        return X, y, record_groups, feature_names, label_info

    ds1 = {str(v) for v in getattr(C, "AAMI_DS1_RECORDS", [])}
    ds2 = {str(v) for v in getattr(C, "AAMI_DS2_RECORDS", [])}
    excluded = {str(v) for v in getattr(C, "AAMI_EXCLUDED_PACED_RECORDS", [])}
    if not ds1 or not ds2:
        raise ValueError("AAMI_DS1_RECORDS and AAMI_DS2_RECORDS must both be configured.")
    overlap = ds1.intersection(ds2)
    if overlap:
        raise ValueError(f"AAMI DS1/DS2 record lists overlap: {sorted(overlap)}")

    record_groups = np.asarray(record_groups).astype(str)
    allowed = (ds1 | ds2) - excluded
    keep = np.isin(record_groups, sorted(allowed))
    missing = sorted(allowed - set(np.unique(record_groups).tolist()))
    if missing:
        raise ValueError(f"AAMI protocol records are missing from the loaded dataset: {missing}")
    if not np.any(keep):
        raise ValueError("AAMI protocol filter removed every sample.")

    filtered_info = dict(label_info)
    filtered_info["ProtocolDatasetFilter"] = {
        "Protocol": "aami_ds1_ds2",
        "InputSamples": int(len(y)),
        "OutputSamples": int(np.sum(keep)),
        "ExcludedRecords": sorted(excluded),
        "DS1Records": sorted(ds1),
        "DS2Records": sorted(ds2),
    }
    return X[keep], y[keep], record_groups[keep], feature_names, filtered_info


def _class_group_support(y: np.ndarray, groups: np.ndarray) -> Dict[int, int]:
    """统计每个类别至少出现于多少个独立 record/group 中。"""
    y = np.asarray(y)
    groups = np.asarray(groups)
    result: Dict[int, int] = {}
    for lab in np.unique(y):
        result[int(lab)] = int(np.unique(groups[y == lab]).size)
    return result


def _maybe_reduce_group_cv_folds(y: np.ndarray, groups: np.ndarray, requested_folds: int) -> int:
    """在稀有类别 record 数不足时自动降低 group_cv 折数。

    例如 L 类只出现在 4 个 record 中，则 5-fold group_cv 必然有一个验证 fold 没有 L。
    与其后续再把 L 类 support=0 的 fold 纳入 Macro-F1，不如提前降到 4 折，
    让每个验证 fold 至少有机会覆盖每个类别。
    """
    requested_folds = int(requested_folds)
    if not bool(getattr(C, "GROUP_CV_AUTO_REDUCE_FOLDS_IF_NEEDED", False)):
        return requested_folds

    support = _class_group_support(y, groups)
    if not support:
        return requested_folds
    feasible = min(requested_folds, min(int(v) for v in support.values()))
    unique_groups = int(np.unique(groups).size)
    feasible = min(feasible, unique_groups)
    feasible = max(2, int(feasible))

    if feasible < requested_folds and bool(getattr(C, "VERBOSE", True)):
        print(
            "[FoldCheck] Auto-reduced FOLDS from "
            f"{requested_folds} to {feasible} because class-to-record support is {support}.",
            flush=True,
        )
    return feasible


def _group_fold_balance_score(splits, y: np.ndarray, folds: int) -> float:
    labels, global_counts = np.unique(y, return_counts=True)
    target = global_counts.astype(float) / float(folds)
    min_support = int(getattr(C, "GROUP_CV_MIN_SUPPORT_PER_CLASS", 1))
    require_each = bool(getattr(C, "GROUP_CV_REQUIRE_EACH_CLASS_IN_VAL", False))
    score = 0.0
    for _, val_idx in splits:
        val_counts = np.asarray([np.sum(y[val_idx] == lab) for lab in labels], dtype=float)
        relative_error = (val_counts - target) / np.maximum(target, 1.0)
        # 使用相对偏差，使稀有类别与多数类别在该统计量中具有相近影响。
        score += float(np.mean(relative_error ** 2))
        # 如果要求每折都有每个类别，则对低于阈值的类别加大惩罚；
        # 即使没有找到完美划分，也会优先选择“缺类最少、缺得最轻”的候选。
        if require_each:
            shortage = np.maximum(0.0, float(min_support) - val_counts)
            score += float(np.sum(shortage)) * 10000.0
        else:
            score += float(np.sum(val_counts <= 0)) * 1000.0
    return score


def _splits_satisfy_group_cv_requirement(splits, y: np.ndarray) -> bool:
    """检查每个验证 fold 是否满足“每类至少 min_support 个样本”的要求。"""
    if not bool(getattr(C, "GROUP_CV_REQUIRE_EACH_CLASS_IN_VAL", False)):
        return True
    labels = np.unique(y)
    min_support = int(getattr(C, "GROUP_CV_MIN_SUPPORT_PER_CLASS", 1))
    for _, val_idx in splits:
        counts = np.asarray([np.sum(y[val_idx] == lab) for lab in labels], dtype=int)
        if np.any(counts < min_support):
            return False
    return True


def make_patient_groups(record_groups: np.ndarray) -> np.ndarray:
    """将 record_id 映射为患者组；未列入别名表的记录各自视为一名患者。"""
    records = np.asarray(record_groups).astype(str)
    aliases = {str(k): str(v) for k, v in (getattr(C, "PATIENT_GROUP_ALIASES", {}) or {}).items()}
    return np.asarray([aliases.get(record, f"patient_{record}") for record in records], dtype=str)


def _partition_balance_score(
    selected_idx: np.ndarray,
    complement_idx: np.ndarray,
    y: np.ndarray,
    target_fraction: float,
    require_all_classes: bool,
) -> float:
    """评价一个患者分组分区的类别与样本量平衡，不使用任何模型输出。"""
    labels, global_counts = np.unique(y, return_counts=True)
    selected_counts = np.asarray([np.sum(y[selected_idx] == lab) for lab in labels], dtype=float)
    expected = global_counts.astype(float) * float(target_fraction)
    relative_error = (selected_counts - expected) / np.maximum(expected, 1.0)
    score = float(np.mean(relative_error ** 2))
    score += float((len(selected_idx) / max(1, len(y)) - target_fraction) ** 2)

    min_samples = max(1, int(getattr(C, "PATIENT_MIN_CLASS_SAMPLES_PER_SPLIT", 1)))
    if require_all_classes:
        complement_counts = np.asarray([np.sum(y[complement_idx] == lab) for lab in labels], dtype=int)
        missing = int(np.sum(selected_counts < min_samples) + np.sum(complement_counts < min_samples))
        score += float(missing) * 1_000_000.0
    return score


def _best_group_kfold_set(
    X, y, groups, n_splits: int, seed: int, n_candidates: int,
    require_all_classes: Optional[bool] = None,
    min_class_samples: Optional[int] = None,
):
    """在患者层搜索类别覆盖完整且患者数均衡的互斥 K 折。"""
    y = np.asarray(y)
    groups = np.asarray(groups).astype(str)
    unique_groups, group_inverse = np.unique(groups, return_inverse=True)
    if unique_groups.size < n_splits:
        raise ValueError(f"Need at least {n_splits} patient groups, got {unique_groups.size}.")
    labels, global_counts = np.unique(y, return_counts=True)
    group_class_counts = np.zeros((unique_groups.size, labels.size), dtype=np.int64)
    group_sizes = np.bincount(group_inverse, minlength=unique_groups.size).astype(np.int64)
    for label_idx, label in enumerate(labels):
        group_class_counts[:, label_idx] = np.bincount(
            group_inverse[y == label], minlength=unique_groups.size
        )

    target_class = global_counts.astype(float) / float(n_splits)
    target_size = float(len(y)) / float(n_splits)
    target_groups = float(unique_groups.size) / float(n_splits)
    require_all = (
        bool(getattr(C, "PATIENT_REQUIRE_ALL_CLASSES_IN_TEST", True))
        if require_all_classes is None else bool(require_all_classes)
    )
    min_samples = max(
        1,
        int(
            getattr(C, "PATIENT_MIN_CLASS_SAMPLES_PER_SPLIT", 1)
            if min_class_samples is None else min_class_samples
        ),
    )
    rng = np.random.default_rng(int(seed))
    best = None
    for candidate_id in range(max(1, int(n_candidates))):
        order = rng.permutation(unique_groups.size)
        fold_order = rng.permutation(int(n_splits))
        assignment = np.empty(unique_groups.size, dtype=int)
        for position, group_idx in enumerate(order):
            assignment[group_idx] = int(fold_order[position % int(n_splits)])

        fold_class = np.vstack([
            np.sum(group_class_counts[assignment == fold_id], axis=0)
            for fold_id in range(int(n_splits))
        ])
        fold_sizes = np.asarray([
            np.sum(group_sizes[assignment == fold_id]) for fold_id in range(int(n_splits))
        ], dtype=float)
        fold_group_counts = np.asarray([
            np.sum(assignment == fold_id) for fold_id in range(int(n_splits))
        ], dtype=float)
        missing = int(np.sum(fold_class < min_samples))
        if require_all and missing:
            continue
        relative_class_error = (fold_class - target_class[None, :]) / np.maximum(target_class[None, :], 1.0)
        score = float(np.mean(relative_class_error ** 2))
        score += float(np.mean(((fold_sizes - target_size) / max(target_size, 1.0)) ** 2))
        score += 0.1 * float(np.mean(((fold_group_counts - target_groups) / max(target_groups, 1.0)) ** 2))
        item = (score, candidate_id, assignment.copy())
        if best is None or item[:2] < best[:2]:
            best = item

    if best is None:
        support = _class_group_support(y, groups)
        raise ValueError(
            f"No patient partition covered every class in all {n_splits} folds after "
            f"{n_candidates} candidates; class-to-patient support={support}."
        )
    assignment = best[2]
    all_indices = np.arange(len(y), dtype=int)
    splits = []
    for fold_id in range(int(n_splits)):
        test_mask = assignment[group_inverse] == fold_id
        splits.append((all_indices[~test_mask], all_indices[test_mask]))
    return splits


def _best_inner_group_split(X, y, groups, development_idx: np.ndarray, outer_fold: int, seed: int):
    dev_idx = np.asarray(development_idx, dtype=int)
    y_dev = np.asarray(y)[dev_idx]
    groups_dev = np.asarray(groups)[dev_idx]
    unique_patients = np.unique(groups_dev).size
    requested = max(2, int(getattr(C, "PATIENT_INNER_FOLDS", 4)))
    class_patient_support = _class_group_support(y_dev, groups_dev)
    min_class_patient_support = min(class_patient_support.values()) if class_patient_support else 0
    inner_folds = min(requested, int(unique_patients), int(min_class_patient_support))
    if inner_folds < 2:
        raise ValueError(
            f"Outer fold {outer_fold} cannot create a patient-disjoint inner train/validation split; "
            f"class-to-patient support={class_patient_support}. Each class must occur in at least two "
            "development patients. Reduce outer FOLDS or revise the patient cohort."
        )
    if inner_folds < requested and bool(getattr(C, "VERBOSE", True)):
        print(
            f"[PatientSplit] Outer Fold {outer_fold}: reduced inner folds from {requested} to "
            f"{inner_folds}; development class-to-patient support={class_patient_support}.",
            flush=True,
        )
    require_all = bool(getattr(C, "PATIENT_REQUIRE_ALL_CLASSES_IN_VALIDATION", True))
    inner_min_samples = max(1, int(getattr(C, "PATIENT_MIN_CLASS_SAMPLES_IN_VALIDATION", 1)))
    n_candidates = max(1, int(getattr(C, "PATIENT_INNER_SPLIT_CANDIDATES", 128)))
    base_seed = int(seed) + int(outer_fold) * 100_003
    inner_splits = _best_group_kfold_set(
        X[dev_idx], y_dev, groups_dev, inner_folds, base_seed, n_candidates,
        require_all_classes=require_all, min_class_samples=inner_min_samples,
    )
    candidates = []
    for inner_fold, (train_local, val_local) in enumerate(inner_splits, start=1):
        score = _partition_balance_score(
            val_local, train_local, y_dev, 1.0 / float(inner_folds), require_all
        )
        candidates.append((float(score), int(inner_fold), train_local, val_local))
    _, _, train_local, val_local = min(candidates, key=lambda item: (item[0], item[1]))
    global_splits = tuple(
        (
            dev_idx[np.asarray(inner_train, dtype=int)],
            dev_idx[np.asarray(inner_val, dtype=int)],
        )
        for inner_train, inner_val in inner_splits
    )
    return (
        dev_idx[np.asarray(train_local, dtype=int)],
        dev_idx[np.asarray(val_local, dtype=int)],
        global_splits,
    )


def _assert_patient_independent_fold(fd: FoldData, patient_groups: np.ndarray, y: np.ndarray) -> None:
    if fd.test_idx is None or len(fd.test_idx) == 0:
        raise ValueError(f"{fd.split_protocol} Fold {fd.fold} has no independent test set.")
    index_sets = {
        "train": set(np.asarray(fd.train_idx, dtype=int).tolist()),
        "validation": set(np.asarray(fd.val_idx, dtype=int).tolist()),
        "test": set(np.asarray(fd.test_idx, dtype=int).tolist()),
    }
    group_sets = {
        name: set(np.asarray(patient_groups)[np.asarray(indices, dtype=int)].tolist())
        for name, indices in {
            "train": fd.train_idx, "validation": fd.val_idx, "test": fd.test_idx
        }.items()
    }
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if index_sets[left].intersection(index_sets[right]):
            raise AssertionError(f"Sample overlap between {left} and {right} in Fold {fd.fold}.")
        overlap = group_sets[left].intersection(group_sets[right])
        if overlap:
            raise AssertionError(
                f"Patient overlap between {left} and {right} in Fold {fd.fold}: {sorted(overlap)}"
            )
    labels = set(np.unique(y).tolist())
    for name, indices in [("train", fd.train_idx), ("validation", fd.val_idx), ("test", fd.test_idx)]:
        present = set(np.unique(y[np.asarray(indices, dtype=int)]).tolist())
        if present != labels:
            missing = sorted(labels - present)
            raise ValueError(f"Fold {fd.fold} {name} split is missing configured classes: {missing}")


def make_nested_patient_group_cv(X, y, patient_groups, folds: int, seed: int) -> List[FoldData]:
    """外层患者测试、内层患者验证的严格嵌套交叉验证。"""
    requested_folds = int(folds)
    support = _class_group_support(np.asarray(y), np.asarray(patient_groups))
    feasible_folds = min(requested_folds, min(support.values()), np.unique(patient_groups).size)
    if feasible_folds < 2:
        raise ValueError(f"Nested patient CV is infeasible; class-to-patient support={support}.")
    if feasible_folds < requested_folds and bool(getattr(C, "VERBOSE", True)):
        print(
            f"[PatientSplit] Reduced outer folds from {requested_folds} to {feasible_folds}; "
            f"class-to-patient support={support}.", flush=True,
        )

    outer_splits = _best_group_kfold_set(
        X, y, patient_groups, feasible_folds, seed,
        int(getattr(C, "PATIENT_OUTER_SPLIT_CANDIDATES", 128)),
    )
    result = []
    for fold_id, (development_idx, test_idx) in enumerate(outer_splits, start=1):
        train_idx, val_idx, inner_splits = _best_inner_group_split(
            X, y, patient_groups, np.asarray(development_idx, dtype=int), fold_id, seed
        )
        fd = FoldData(
            fold=fold_id,
            train_idx=np.asarray(train_idx, dtype=int),
            val_idx=np.asarray(val_idx, dtype=int),
            test_idx=np.asarray(test_idx, dtype=int),
            split_protocol="nested_patient_group_cv",
            inner_splits=inner_splits,
        )
        _assert_patient_independent_fold(fd, patient_groups, y)
        result.append(fd)
    return result


def make_aami_ds1_ds2_split(X, y, patient_groups, record_groups, seed: int) -> List[FoldData]:
    """构建 AAMI DS1 开发 / DS2 测试，并在 DS1 内建立患者独立验证集。"""
    records = np.asarray(record_groups).astype(str)
    patients = np.asarray(patient_groups).astype(str)
    ds1 = {str(v) for v in getattr(C, "AAMI_DS1_RECORDS", [])}
    ds2 = {str(v) for v in getattr(C, "AAMI_DS2_RECORDS", [])}
    development_mask = np.isin(records, sorted(ds1))
    test_mask = np.isin(records, sorted(ds2))

    overlap_patients = set(patients[development_mask]).intersection(set(patients[test_mask]))
    strict = bool(getattr(C, "AAMI_STRICT_PATIENT_ISOLATION", True))
    if overlap_patients and strict:
        policy = str(getattr(C, "AAMI_PATIENT_OVERLAP_POLICY", "move_test_to_development")).lower()
        if policy != "move_test_to_development":
            raise ValueError("AAMI_PATIENT_OVERLAP_POLICY must be 'move_test_to_development'.")
        move_mask = test_mask & np.isin(patients, sorted(overlap_patients))
        development_mask = development_mask | move_mask
        test_mask = test_mask & ~move_mask
    elif overlap_patients:
        print(
            f"[PatientSplit] WARNING: exact AAMI lists share patients {sorted(overlap_patients)}; "
            "this is standard-list reproduction, not strict patient isolation.", flush=True,
        )

    development_idx = np.flatnonzero(development_mask)
    test_idx = np.flatnonzero(test_mask)
    train_idx, val_idx, inner_splits = _best_inner_group_split(
        X, y, patients, development_idx, outer_fold=1, seed=seed
    )
    protocol_name = "aami_ds1_ds2_strict" if strict else "aami_ds1_ds2"
    fd = FoldData(
        1, train_idx, val_idx, test_idx=test_idx,
        split_protocol=protocol_name, inner_splits=inner_splits,
    )
    if strict:
        _assert_patient_independent_fold(fd, patients, y)
    return [fd]


def _split_protocol_name() -> str:
    """规范化面向用户显示的数据划分协议名称。"""
    value = str(getattr(C, "SPLIT_PROTOCOL", "group_cv") or "group_cv").strip().lower()
    aliases = {
        "grouped_cv": "group_cv",
        "group_kfold": "group_cv",
        "record_cv": "group_cv",
        "nested_group_cv": "nested_patient_group_cv",
        "patient_group_cv": "nested_patient_group_cv",
        "patient_cv": "nested_patient_group_cv",
        "aami": "aami_ds1_ds2",
        "ds1_ds2": "aami_ds1_ds2",
        "random_811": "senior_random_811",
        "holdout_811": "senior_random_811",
        "senior": "senior_random_811",
        "frtcn": "senior_random_811",
        "repeated_811": "repeated_random_811",
        "repeated_holdout_811": "repeated_random_811",
        "random_811_repeated": "repeated_random_811",
        "multi_random_811": "repeated_random_811",
    }
    return aliases.get(value, value)


def _make_random_811_fold(X, y, groups, split_seed: int, fold_id: int, protocol_name: str) -> FoldData:
    """构建一个 beat-level 随机 8:1:1 hold-out fold。

    该函数是 senior_random_811 与 repeated_random_811 的共同实现。
    注意：groups 只用于日志统计，不参与划分；因此同一 record 的心拍可能同时出现在
    train / validation / test 中。
    """
    test_ratio = float(getattr(C, "HOLDOUT_TEST_RATIO", 0.10))
    val_ratio = float(getattr(C, "HOLDOUT_VAL_RATIO", 0.10))
    stratify_enabled = bool(getattr(C, "HOLDOUT_STRATIFY", False))

    if not (0.0 < test_ratio < 1.0):
        raise ValueError("HOLDOUT_TEST_RATIO must be in (0, 1).")
    if not (0.0 < val_ratio < 1.0 - test_ratio):
        raise ValueError("HOLDOUT_VAL_RATIO must be in (0, 1 - HOLDOUT_TEST_RATIO).")

    indices = np.arange(len(y), dtype=int)
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)
    stratify_all = y_arr if stratify_enabled else None

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=int(split_seed),
        shuffle=True,
        stratify=stratify_all,
    )

    # 这里等价于 val_ratio / (1 - test_ratio)，与师兄 utils.py 逻辑一致。
    val_relative_ratio = val_ratio / (1.0 - test_ratio)
    stratify_train_val = y_arr[train_val_idx] if stratify_enabled else None
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative_ratio,
        random_state=int(split_seed),
        shuffle=True,
        stratify=stratify_train_val,
    )

    if bool(getattr(C, "VERBOSE", True)):
        print(
            f"[Split] {protocol_name}: Fold {int(fold_id)} beat-level random 8:1:1 "
            f"(seed={int(split_seed)}, stratify={stratify_enabled}).",
            flush=True,
        )
        for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            labels, counts = np.unique(y_arr[idx], return_counts=True)
            dist = {int(label): int(count) for label, count in zip(labels, counts)}
            print(
                f"[Split] F{int(fold_id)} {name}: n={len(idx)}, "
                f"records={len(np.unique(groups_arr[idx]))}, class_counts={dist}",
                flush=True,
            )

    return FoldData(
        fold=int(fold_id),
        train_idx=np.asarray(train_idx, dtype=int),
        val_idx=np.asarray(val_idx, dtype=int),
        test_idx=np.asarray(test_idx, dtype=int),
        split_protocol=str(protocol_name),
    )


def make_senior_random_811_split(X, y, groups) -> List[FoldData]:
    """复现师兄 FRTCN 源码的单次随机 8:1:1 划分。

    Source-equivalent logic:
        train_val, test = train_test_split(all_beats, test_size=0.1, random_state=1)
        train, val = train_test_split(train_val, test_size=0.1/(1-0.1), random_state=1)

    该模式固定只返回 Fold=1。若需要多个随机 8:1:1 划分，请使用
    SPLIT_PROTOCOL="repeated_random_811"。
    """
    split_seed = int(getattr(C, "HOLDOUT_RANDOM_SEED", 1))
    if bool(getattr(C, "VERBOSE", True)):
        print(
            "[Split] WARNING: senior_random_811 ignores record groups by design. "
            "Use it for senior-code comparison, not record-independent validation.",
            flush=True,
        )
    return [_make_random_811_fold(X, y, groups, split_seed=split_seed, fold_id=1, protocol_name="senior_random_811")]


def make_repeated_random_811_splits(X, y, groups, folds: int, seed: int) -> List[FoldData]:
    """生成多个独立随机 8:1:1 hold-out 划分。

    与 group_cv 不同，这里的 Fold 不是互斥 K 折交叉验证，而是 repeated hold-out：
        Fold 1 使用 seed = HOLDOUT_RANDOM_SEED
        Fold 2 使用 seed = HOLDOUT_RANDOM_SEED + HOLDOUT_RANDOM_SEED_STEP
        ...
    每个 fold 都重新随机划分 train / validation / test，适合 Accuracy-first 场景下
    用多个随机划分评估结果稳定性。
    """
    n_folds = int(folds)
    if n_folds < 1:
        raise ValueError("FOLDS must be >= 1 for repeated_random_811.")
    base_seed = int(getattr(C, "HOLDOUT_RANDOM_SEED", seed))
    step = int(getattr(C, "HOLDOUT_RANDOM_SEED_STEP", 1009))
    if step == 0:
        raise ValueError("HOLDOUT_RANDOM_SEED_STEP must be non-zero for repeated_random_811.")

    if bool(getattr(C, "VERBOSE", True)):
        print(
            f"[Split] repeated_random_811: generating {n_folds} independent beat-level "
            f"8:1:1 hold-out folds; base_seed={base_seed}, seed_step={step}.",
            flush=True,
        )
        print(
            "[Split] WARNING: record groups are intentionally ignored. "
            "This protocol targets Accuracy-first / TCN-style comparison, not record-independent validation.",
            flush=True,
        )

    fold_data = []
    used_seeds = []
    for fold_id in range(1, n_folds + 1):
        split_seed = base_seed + (fold_id - 1) * step
        used_seeds.append(int(split_seed))
        fold_data.append(
            _make_random_811_fold(
                X,
                y,
                groups,
                split_seed=split_seed,
                fold_id=fold_id,
                protocol_name="repeated_random_811",
            )
        )

    if bool(getattr(C, "VERBOSE", True)):
        print(f"[Split] repeated_random_811 seeds: {used_seeds}", flush=True)
    return fold_data


def make_folds(X, y, groups, folds: int, seed: int, record_groups=None) -> List[FoldData]:
    """构建数据划分。

    group_cv 模式下会优先使用 StratifiedGroupKFold，并在多个随机候选中选择类别分布最均衡的一组。
    如果某个类别出现的 record 数少于配置的 FOLDS，可按配置自动降折，避免天然缺类 fold。
    """
    split_protocol = _split_protocol_name()
    if split_protocol == "nested_patient_group_cv":
        return make_nested_patient_group_cv(X, y, groups, folds=folds, seed=seed)
    if split_protocol == "aami_ds1_ds2":
        if record_groups is None:
            raise ValueError("aami_ds1_ds2 requires original record_groups in addition to patient groups.")
        return make_aami_ds1_ds2_split(X, y, groups, record_groups, seed=seed)
    if split_protocol == "senior_random_811":
        return make_senior_random_811_split(X, y, groups)
    if split_protocol == "repeated_random_811":
        return make_repeated_random_811_splits(X, y, groups, folds=folds, seed=seed)
    if split_protocol != "group_cv":
        raise ValueError(
            "SPLIT_PROTOCOL must be 'nested_patient_group_cv', 'aami_ds1_ds2', "
            "'group_cv', 'senior_random_811', or "
            f"'repeated_random_811'. Got {split_protocol!r}."
        )

    unique_groups = np.unique(groups)
    folds = _maybe_reduce_group_cv_folds(np.asarray(y), np.asarray(groups), int(folds))
    if unique_groups.size < folds:
        folds = max(2, int(unique_groups.size))
        if bool(getattr(C, "VERBOSE", True)):
            print(f"[FoldCheck] Reduced folds to {folds} because only {unique_groups.size} groups are available.", flush=True)

    if unique_groups.size >= folds and StratifiedGroupKFold is not None:
        n_candidates = max(1, int(getattr(C, "GROUP_FOLD_SEARCH_CANDIDATES", 96)))
        candidates = []
        valid_candidates = []
        for offset in range(n_candidates):
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=int(seed) + offset)
            splits = list(splitter.split(X, y, groups))
            item = (_group_fold_balance_score(splits, y, folds), splits)
            candidates.append(item)
            if _splits_satisfy_group_cv_requirement(splits, y):
                valid_candidates.append(item)
        pool = valid_candidates if valid_candidates else candidates
        _, splits = min(pool, key=lambda item: item[0])
        split_name = f"StratifiedGroupKFold best-of-{n_candidates}"
        if bool(getattr(C, "GROUP_CV_REQUIRE_EACH_CLASS_IN_VAL", False)):
            split_name += f"; valid_candidates={len(valid_candidates)}"
            if not valid_candidates and bool(getattr(C, "VERBOSE", True)):
                print(
                    "[FoldCheck] WARNING: no candidate split fully satisfied "
                    "GROUP_CV_REQUIRE_EACH_CLASS_IN_VAL; using the least-imbalanced split.",
                    flush=True,
                )
    elif unique_groups.size >= folds:
        splitter = GroupKFold(n_splits=folds)
        splits = list(splitter.split(X, y, groups))
        split_name = "GroupKFold fallback"
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y))
        split_name = "StratifiedKFold fallback (no usable groups)"

    fold_data = [FoldData(i + 1, tr, va, test_idx=None, split_protocol="group_cv") for i, (tr, va) in enumerate(splits)]
    if bool(getattr(C, "VERBOSE", True)):
        labels = np.unique(y)
        warn_min = int(getattr(C, "GROUP_FOLD_MIN_VALIDATION_COUNT_WARN", 5))
        print(f"[FoldCheck] {split_name}; unique groups={unique_groups.size}; effective folds={folds}", flush=True)
        print(f"[FoldCheck] class-to-record support: {_class_group_support(np.asarray(y), np.asarray(groups))}", flush=True)
        for fd in fold_data:
            counts = {int(lab): int(np.sum(y[fd.val_idx] == lab)) for lab in labels}
            rare = [lab for lab, count in counts.items() if count < warn_min]
            suffix = f"  WARNING: low validation support for classes {rare}" if rare else ""
            print(f"[FoldCheck] Fold {fd.fold} validation class counts: {counts}{suffix}", flush=True)
    return fold_data


def safe_corr_abs(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3:
        return np.eye(X.shape[1])
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    return corr


def mrmr_ranking(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """使用互信息相关性与绝对相关系数冗余项实现简洁的贪心 mRMR 排序。"""
    dim = X.shape[1]
    try:
        relevance = mutual_info_classif(X, y, discrete_features=False, random_state=seed)
    except Exception:
        fvals, _ = f_classif(X, y)
        relevance = np.nan_to_num(fvals, nan=0.0, posinf=0.0, neginf=0.0)
    relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)
    corr = safe_corr_abs(X)

    selected = []
    remaining = set(range(dim))
    while remaining:
        best_j, best_score = None, -np.inf
        for j in remaining:
            redundancy = 0.0 if not selected else float(np.mean(corr[j, selected]))
            score = float(relevance[j] - C.MRMR_REDUNDANCY_WEIGHT * redundancy)
            if score > best_score:
                best_score = score
                best_j = j
        selected.append(best_j)
        remaining.remove(best_j)
    return np.asarray(selected, dtype=int)


def mrmr_soft_prior_scores(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Build a training-only [0, 1] soft prior from the mRMR ranking.

    The full feature space remains searchable. The prior only changes the
    inner patient-CV objective, so it cannot use outer test-patient labels.
    """
    dim = int(np.asarray(X).shape[1])
    ranking = mrmr_ranking(X, y, seed=seed)
    scores = np.zeros(dim, dtype=np.float64)
    if dim <= 1:
        scores[:] = 1.0
    else:
        scores[np.asarray(ranking, dtype=int)] = np.linspace(1.0, 0.0, dim)
    return scores


def mrmr_multiscale_seed_masks(ranking: np.ndarray, dim: int, seed: int) -> list:
    """Build diverse, training-only MRMR seed masks for ST-FODM-MRMR-PATIENT."""
    ranking = np.asarray(ranking, dtype=int).reshape(-1)
    if ranking.size != int(dim):
        raise ValueError("MRMR ranking length must match the optimization dimension.")
    ratios = [float(v) for v in (getattr(C, "ST_MRMR_SEED_TOPK_RATIOS", [0.10, 0.20, 0.30, 0.40]) or [])]
    desired = max(1, int(round(int(C.NPOP) * float(getattr(C, "ST_MRMR_SEED_RATIO", 0.60)))))
    rng = np.random.default_rng(int(seed))
    masks = []
    for i in range(desired):
        ratio = ratios[i % len(ratios)] if ratios else 0.20
        k = min(int(dim), max(int(C.MIN_FEATURES), int(round(float(dim) * ratio))))
        mask = np.zeros(int(dim), dtype=bool)
        mask[ranking[:k]] = True
        # Retain the MRMR scale while producing related but non-identical seeds.
        swaps = int(round(k * float(getattr(C, "ST_MRMR_SEED_PERTURB_RATIO", 0.08))))
        if swaps > 0 and k < int(dim):
            remove = rng.choice(np.flatnonzero(mask), size=min(swaps, k), replace=False)
            add = rng.choice(np.flatnonzero(~mask), size=min(swaps, int(dim) - k), replace=False)
            mask[remove] = False
            mask[add] = True
        masks.append(mask)
    return masks


def _json_safe_label(label):
    """将 NumPy 标量标签转换为可被 JSON 序列化的 Python 标量。"""
    if isinstance(label, np.generic):
        return label.item()
    return label


def _resolve_positive_label(labels):
    """确定二分类敏感度指标使用的正类。"""
    labels = list(labels)
    configured = getattr(C, "POSITIVE_LABEL", None)
    if configured is not None and configured in labels:
        return configured
    # 默认取排序后较大的标签，与 0/1 正常/异常任务一致。
    return labels[-1]


def classification_metrics(y_true, y_pred, labels=None) -> dict:
    """计算分类指标，并支持 group_cv 下“零 support 类别不参与宏平均”。

    为什么需要这个开关：
        record-level group_cv 会按病人/记录分组。有些稀有类（例如 L）可能只出现在少数
        record 中，因此某个验证 fold 里可能完全没有该类样本。此时如果仍把该类 F1=0
        纳入 Macro-F1，会把“验证集中不存在该类”误当成“模型该类预测失败”。

    METRIC_IGNORE_ZERO_SUPPORT_CLASSES=True 时：
        1. 混淆矩阵仍保留完整 labels，保证 N/A/V/L/R 维度一致；
        2. ClassMetrics 逐类别表仍保留 support=0 的类别，便于审计；
        3. MacroF1 / MacroPrecision / MacroRecall / Sensitivity / Specificity 只在
           y_true 中实际出现的类别上求平均。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    else:
        labels = list(labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = float(cm.sum())
    prec, rec, f1v, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    ignore_zero_support = bool(getattr(C, "METRIC_IGNORE_ZERO_SUPPORT_CLASSES", False))
    if ignore_zero_support:
        metric_idx = np.flatnonzero(np.asarray(sup) > 0)
        if metric_idx.size == 0:
            metric_idx = np.arange(len(labels), dtype=int)
    else:
        metric_idx = np.arange(len(labels), dtype=int)

    metric_idx_set = set(int(i) for i in metric_idx.tolist())
    metric_labels = [labels[int(i)] for i in metric_idx]
    ignored_labels = [labels[int(i)] for i in range(len(labels)) if int(i) not in metric_idx_set]

    macro_f1 = float(np.mean(f1v[metric_idx])) if metric_idx.size else 0.0
    macro_precision = float(np.mean(prec[metric_idx])) if metric_idx.size else 0.0
    macro_recall = float(np.mean(rec[metric_idx])) if metric_idx.size else 0.0
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_acc = macro_recall

    per_class_specificity = []
    per_class_rows = []
    for i, lab in enumerate(labels):
        tp_i = float(cm[i, i])
        fp_i = float(cm[:, i].sum() - tp_i)
        fn_i = float(cm[i, :].sum() - tp_i)
        tn_i = float(total - tp_i - fp_i - fn_i)
        spec_i = float(tn_i / (tn_i + fp_i + EPS))
        per_class_specificity.append(spec_i)
        per_class_rows.append({
            "Label": _json_safe_label(lab),
            "Precision": float(prec[i]),
            "Recall": float(rec[i]),
            "Sensitivity": float(rec[i]),
            "Specificity": spec_i,
            "F1": float(f1v[i]),
            "Support": int(sup[i]),
            "IncludedInMacro": bool(i in metric_idx_set),
        })

    positive_label = None
    if len(labels) == 2:
        positive_label = _resolve_positive_label(labels)
        pos_idx = labels.index(positive_label)
        tp = float(cm[pos_idx, pos_idx])
        fn = float(cm[pos_idx, :].sum() - tp)
        fp = float(cm[:, pos_idx].sum() - tp)
        tn = float(total - tp - fn - fp)
        sensitivity = float(tp / (tp + fn + EPS))
        specificity = float(tn / (tn + fp + EPS))
        precision_pos = float(tp / (tp + fp + EPS))
        f1_positive = float(2.0 * precision_pos * sensitivity / (precision_pos + sensitivity + EPS))
        neg_recall = specificity
        neg_precision = float(tn / (tn + fn + EPS))
        f1_negative = float(2.0 * neg_precision * neg_recall / (neg_precision + neg_recall + EPS))
    else:
        tp = tn = fp = fn = np.nan
        f1_positive = np.nan
        f1_negative = np.nan
        sensitivity = macro_recall
        specificity_values = np.asarray(per_class_specificity, dtype=float)
        specificity = float(np.mean(specificity_values[metric_idx])) if metric_idx.size else np.nan

    return {
        "MacroF1": macro_f1,
        "WeightedF1": weighted_f1,
        "MacroPrecision": macro_precision,
        "MacroRecall": macro_recall,
        "Accuracy": accuracy,
        "BalancedAccuracy": balanced_acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "F1Positive": f1_positive,
        "F1Negative": f1_negative,
        "PositiveLabel": _json_safe_label(positive_label) if positive_label is not None else "macro",
        "ConfusionMatrix": cm.tolist(),
        "ClassMetrics": per_class_rows,
        "MetricLabels": [_json_safe_label(v) for v in metric_labels],
        "IgnoredZeroSupportLabels": [_json_safe_label(v) for v in ignored_labels],
        "IgnoreZeroSupportClasses": bool(ignore_zero_support),
    }


def _patient_balanced_macro_f1(y_true, y_pred, patient_groups, labels) -> float:
    """Compute Macro-F1 per validation patient, then average patients equally."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    groups = np.asarray(patient_groups).astype(str)
    if y_true.size == 0 or y_true.size != y_pred.size or y_true.size != groups.size:
        return np.nan
    values = []
    for group in np.unique(groups):
        keep = groups == group
        present = [int(label) for label in labels if np.any(y_true[keep] == int(label))]
        if present:
            values.append(float(f1_score(
                y_true[keep], y_pred[keep], labels=present,
                average="macro", zero_division=0,
            )))
    return float(np.mean(values)) if values else np.nan


def _aggregate_inner_fold_metrics(
    fold_metrics: list,
    selected_ratio: float,
    macro_f1_std_penalty: float = 0.0,
    patient_macro_weight: float = 0.0,
) -> dict:
    """按内折等权聚合指标，避免大记录患者主导适应度。"""
    if not fold_metrics:
        raise ValueError("At least one inner-fold metric dictionary is required.")
    # _fitness_from_metrics 会补齐 MinorityF1 / HardF1 / WorstMinorityF1 等字段，
    # 必须先逐折调用，再读取这些指标做聚合。
    fold_fitness = []
    patient_weight = min(1.0, max(0.0, float(patient_macro_weight)))
    for metric in fold_metrics:
        base_fitness = _fitness_from_metrics(metric, selected_ratio)
        base_utility = float(metric.get("FitnessUtility", 0.0))
        patient_macro = float(metric.get("PatientMacroF1", np.nan))
        if patient_weight > 0.0 and np.isfinite(patient_macro):
            utility = (1.0 - patient_weight) * base_utility + patient_weight * patient_macro
            metric["PatientRobustFitnessUtility"] = float(utility)
            fold_fitness.append(float(1.0 - utility + float(C.LAMBDA_PENALTY) * selected_ratio))
        else:
            metric["PatientRobustFitnessUtility"] = float(base_utility)
            fold_fitness.append(float(base_fitness))
    scalar_keys = [
        "MacroF1", "WeightedF1", "MacroPrecision", "MacroRecall", "Accuracy",
        "BalancedAccuracy", "Sensitivity", "Specificity", "F1Positive",
        "F1Negative", "MinorityF1", "HardF1", "WorstMinorityF1",
        "FitnessUtility", "PatientMacroF1", "PatientRobustFitnessUtility",
    ]
    out = {}
    for key in scalar_keys:
        values = np.asarray([m.get(key, np.nan) for m in fold_metrics], dtype=float)
        finite = values[np.isfinite(values)]
        out[key] = float(np.mean(finite)) if finite.size else np.nan
    cms = [np.asarray(m.get("ConfusionMatrix", []), dtype=np.int64) for m in fold_metrics]
    if cms and all(cm.shape == cms[0].shape for cm in cms):
        out["ConfusionMatrix"] = np.sum(cms, axis=0).tolist()
    else:
        out["ConfusionMatrix"] = []
    out["ClassMetrics"] = fold_metrics[0].get("ClassMetrics", [])
    out["MetricLabels"] = fold_metrics[0].get("MetricLabels", [])
    out["InnerCVFoldMetrics"] = fold_metrics
    macro_values = np.asarray([m.get("MacroF1", np.nan) for m in fold_metrics], dtype=float)
    finite_macro_values = macro_values[np.isfinite(macro_values)]
    macro_std = float(np.std(finite_macro_values, ddof=0)) if finite_macro_values.size else np.nan
    mean_fold_fitness = float(np.mean(fold_fitness))
    penalty_weight = max(0.0, float(macro_f1_std_penalty))
    stability_penalty = float(penalty_weight * macro_std) if np.isfinite(macro_std) else 0.0
    out["InnerCVFolds"] = int(len(fold_metrics))
    out["InnerCVAggregation"] = "unweighted_mean_across_patient_group_folds"
    # 非 RobustCV 方法传入 0，完全保留既有方法的原始平均 fitness。
    out["InnerCVMeanFitness"] = mean_fold_fitness
    out["PatientMacroF1Weight"] = patient_weight
    out["InnerCVMacroF1Std"] = macro_std
    out["InnerCVMacroF1StdPenaltyWeight"] = penalty_weight
    out["InnerCVMacroF1StdPenalty"] = stability_penalty
    out["Fitness"] = float(mean_fold_fitness + stability_penalty)
    out["FinalBest"] = float(out["Fitness"])
    return out


class BLSFeatureSelectionObjective:
    def __init__(
        self, X_train, y_train, X_val, y_val, feature_names, seed: int,
        bls_seed: Optional[int] = None, validation_folds: Optional[list] = None,
        macro_f1_std_penalty: float = 0.0,
        patient_macro_weight: float = 0.0,
        mrmr_prior_enabled: Optional[bool] = None,
        mrmr_objective_lambda: Optional[float] = None,
    ):
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=int)
        self.X_val = np.asarray(X_val, dtype=np.float32)
        self.y_val = np.asarray(y_val, dtype=int)
        self.feature_names = np.asarray(feature_names, dtype=str)
        self.dim = self.X_train.shape[1]
        raw_folds = list(validation_folds or [])
        if raw_folds:
            self.validation_folds = []
            for i, raw_item in enumerate(raw_folds):
                item = dict(raw_item)
                item["fold"] = int(item.get("fold", i + 1))
                item["y_train"] = np.asarray(item["y_train"], dtype=int)
                item["y_val"] = np.asarray(item["y_val"], dtype=int)
                if "val_patient_groups" in item:
                    item["val_patient_groups"] = np.asarray(item["val_patient_groups"]).astype(str)
                if "X_source" in item:
                    item["X_source"] = np.asarray(item["X_source"], dtype=np.float32)
                    item["train_idx"] = np.asarray(item["train_idx"], dtype=int)
                    item["val_idx"] = np.asarray(item["val_idx"], dtype=int)
                    item["scale_mean"] = np.asarray(item["scale_mean"], dtype=np.float32)
                    item["scale_scale"] = np.asarray(item["scale_scale"], dtype=np.float32)
                else:
                    item["X_train"] = np.asarray(item["X_train"], dtype=np.float32)
                    item["X_val"] = np.asarray(item["X_val"], dtype=np.float32)
                self.validation_folds.append(item)
        else:
            self.validation_folds = [{
                "fold": 1, "X_train": self.X_train, "y_train": self.y_train,
                "X_val": self.X_val, "y_val": self.y_val,
            }]
        self.inner_cv_enabled = len(self.validation_folds) > 1
        self.validation_y = np.concatenate([
            item["y_val"] for item in self.validation_folds
        ]).astype(int)
        # seed 控制优化器随机轨迹；同一 Fold × Run 内各方法共享 bls_seed，
        # 避免把 BLS 随机映射差异误判为优化算法优势。
        self.seed = int(seed)
        self.bls_seed = int(seed if bls_seed is None else bls_seed)
        self.macro_f1_std_penalty = max(0.0, float(macro_f1_std_penalty))
        self.patient_macro_weight = min(1.0, max(0.0, float(patient_macro_weight)))
        self.cache: Dict[bytes, Tuple[float, dict]] = {}
        self.evaluated_masks: Dict[bytes, np.ndarray] = {}
        self.model_fit_count = 0
        self.actual_bls_fit_count = 0
        self.call_count = 0
        self.threshold = float(getattr(C, "THRESHOLD", 0.5))
        self.min_features = int(getattr(C, "MIN_FEATURES", 1))
        self.budget_count_mode = str(getattr(C, "BUDGET_COUNT_MODE", "effective_model_fits"))
        raw_mult = max(1, int(getattr(C, "RAW_EVAL_CALL_MULTIPLIER", 8)))
        self.raw_eval_call_limit = max(int(getattr(C, "MAXFES", 1)), int(getattr(C, "MAXFES", 1)) * raw_mult)
        self.eval_history: List[dict] = []
        self.best_eval_fitness = np.inf
        self.best_eval_metrics: dict = {}
        self.best_eval_mask: Optional[np.ndarray] = None
        self.sample_weight = _compute_bls_sample_weight(self.y_train)
        # The fair ablation controls this per method.  The explicit argument
        # prevents a global lambda from silently changing a baseline objective.
        self.mrmr_prior_enabled = (
            bool(getattr(C, "MRMR_OBJECTIVE_ENABLED", False))
            if mrmr_prior_enabled is None else bool(mrmr_prior_enabled)
        )
        self.mrmr_objective_lambda = max(
            0.0,
            float(
                getattr(C, "MRMR_OBJECTIVE_LAMBDA", 0.0)
                if mrmr_objective_lambda is None else mrmr_objective_lambda
            ),
        )
        self.mrmr_prior_scores = np.zeros(self.dim, dtype=np.float64)
        if self.mrmr_prior_enabled:
            self.mrmr_prior_scores = mrmr_soft_prior_scores(
                self.X_train, self.y_train, seed=self.seed
            )

    def vector_to_mask(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        mask = x > float(C.THRESHOLD)
        if int(mask.sum()) < int(C.MIN_FEATURES):
            order = np.argsort(x)[::-1]
            mask[:] = False
            mask[order[: int(C.MIN_FEATURES)]] = True
        return mask

    @staticmethod
    def mask_key(mask: np.ndarray) -> bytes:
        return np.packbits(mask.astype(np.uint8)).tobytes()


    def _record_eval(self, fitness: float, metrics: dict, mask: np.ndarray, cached: bool = False):
        """使用统一字段记录每次 objective 调用。

        历史记录放在 objective 层，而不是分散到各优化器内部，确保所有方法
        使用同一套收敛曲线口径。
        """
        if fitness < self.best_eval_fitness:
            self.best_eval_fitness = float(fitness)
            self.best_eval_metrics = dict(metrics)
            self.best_eval_mask = np.asarray(mask, dtype=bool).copy()
            improved = True
        else:
            improved = False
        best_m = self.best_eval_metrics or {}
        rec = {
            # RawEvalCall 统计优化器发起的全部 objective 请求；
            # EffectiveFEval / ModelFits 只统计真实 BLS 训练。
            # 缓存命中仅用于调试，不应作为论文收敛曲线的正式横轴。
            "EvalCall": int(self.call_count),
            "RawEvalCall": int(self.call_count),
            "EffectiveFEval": int(self.model_fit_count),
            "ModelFits": int(self.model_fit_count),
            "ActualInnerBLSFits": int(self.actual_bls_fit_count),
            "Cached": bool(cached),
            "CurrentFitness": float(fitness),
            "CurrentMacroF1": float(metrics.get("MacroF1", np.nan)),
            # 患者等权 Macro-F1 只来自当前外层开发集的内层验证患者，
            # 用于追踪 MRMR-PATIENT 的搜索目标，而非报告外层测试成绩。
            "CurrentPatientMacroF1": float(metrics.get("PatientMacroF1", np.nan)),
            "CurrentPatientRobustFitnessUtility": float(metrics.get("PatientRobustFitnessUtility", np.nan)),
            "CurrentAccuracy": float(metrics.get("Accuracy", np.nan)),
            "CurrentBalancedAccuracy": float(metrics.get("BalancedAccuracy", np.nan)),
            "CurrentMinorityF1": float(metrics.get("MinorityF1", np.nan)),
            "CurrentHardF1": float(metrics.get("HardF1", np.nan)),
            "CurrentWorstMinorityF1": float(metrics.get("WorstMinorityF1", np.nan)),
            "CurrentSensitivity": float(metrics.get("Sensitivity", np.nan)),
            "CurrentSpecificity": float(metrics.get("Specificity", np.nan)),
            "CurrentSelectedCount": int(metrics.get("SelectedCount", int(np.asarray(mask).sum()))),
            "CurrentSelectedRatio": float(metrics.get("SelectedRatio", np.asarray(mask).sum() / max(1, self.dim))),
            "BestFitness": float(self.best_eval_fitness),
            "BestPatientMacroF1": float(best_m.get("PatientMacroF1", np.nan)),
            "BestPatientRobustFitnessUtility": float(best_m.get("PatientRobustFitnessUtility", np.nan)),
            "BestValMacroF1": float(best_m.get("MacroF1", np.nan)),
            "BestAccuracy": float(best_m.get("Accuracy", np.nan)),
            "BestBalancedAccuracy": float(best_m.get("BalancedAccuracy", np.nan)),
            "BestMinorityF1": float(best_m.get("MinorityF1", np.nan)),
            "BestHardF1": float(best_m.get("HardF1", np.nan)),
            "BestWorstMinorityF1": float(best_m.get("WorstMinorityF1", np.nan)),
            "BestSensitivity": float(best_m.get("Sensitivity", np.nan)),
            "BestSpecificity": float(best_m.get("Specificity", np.nan)),
            "BestSelectedCount": int(best_m.get("SelectedCount", int(self.best_eval_mask.sum()) if self.best_eval_mask is not None else 0)),
            "BestSelectedRatio": float(best_m.get("SelectedRatio", (self.best_eval_mask.sum() / max(1, self.dim)) if self.best_eval_mask is not None else 0.0)),
            "Improved": bool(improved),
        }
        self.eval_history.append(rec)

    def score_mask(self, mask: np.ndarray, record: bool = True, include_predictions: bool = False) -> Tuple[float, dict]:
        """评价一个二进制特征 mask。

        参数说明
        --------
        mask:
            布尔特征选择向量。
        record:
            True 表示这是优化器预算内的一次 objective 调用，需要计入 EvalCalls / ModelFits，
            并写入收敛历史；False 通常只用于最终 best_x 的指标补全，不计入预算。
        include_predictions:
            True 时额外返回 y_true / y_pred / raw score，用于最终保存 predictions/*.npz；
            False 时只返回标量指标和 mask 信息，用于优化过程。

        重要：
            早期版本在每一次 fitness 评价时都把 y_true、y_pred、ScoreMatrix 存进 cache。
            对于 MAXFES=1000 的重任务，这会把每个任务的缓存放大到数百 MB 甚至更高，
            多进程并行时容易造成内存占满和磁盘 pagefile 满载。

            当前版本只在最终结果保存阶段 include_predictions=True，优化过程和 cache
            均保留轻量指标，从根本上降低内存与磁盘压力。
        """
        mask = np.asarray(mask, dtype=bool)
        key = self.mask_key(mask)
        if record:
            self.call_count += 1

        # 优化过程只需要轻量 cache。若最终需要预测数组，则重新拟合一次 best mask，
        # 不计入正式预算，也不写入 history。
        if (not include_predictions) and key in self.cache:
            f_cached, metrics_cached = self.cache[key]
            if record:
                self._record_eval(float(f_cached), metrics_cached, mask, cached=True)
            return f_cached, metrics_cached

        selected_count = int(mask.sum())
        selected_ratio = selected_count / float(self.dim)
        if selected_count <= 0:
            fitness = 1e9
            metrics = {
                "Fitness": fitness,
                "MacroF1": 0.0,
                "Accuracy": 0.0,
                "BalancedAccuracy": 0.0,
                "Sensitivity": 0.0,
                "Specificity": np.nan,
                "SelectedCount": 0,
                "SelectedRatio": 0.0,
                "SelectedMask": mask.astype(int).tolist(),
                "SelectedFeatureNames": [],
            }
            if include_predictions:
                metrics.update({
                    "PredLabels": [],
                    "TrueLabels": self.y_val.astype(int).tolist(),
                    "ScoreMatrix": None,
                })
            if record:
                self._record_eval(float(fitness), metrics, mask, cached=False)
            return fitness, metrics

        selected_indices = np.flatnonzero(mask)
        labels = sorted(np.unique(np.concatenate([
            np.concatenate([item["y_train"] for item in self.validation_folds]),
            self.validation_y,
        ])))
        fold_metrics = []
        all_pred = []
        all_true = []
        all_scores = []
        for item in self.validation_folds:
            X_inner_train, X_inner_val = _inner_fold_selected_arrays(item, mask)
            model = _make_bls_model(
                random_state=self.bls_seed,
                feature_indices=selected_indices,
                input_dim=self.dim,
            )
            fold_weight = _compute_bls_sample_weight(item["y_train"])
            model.fit(
                X_inner_train, item["y_train"],
                sample_weight=fold_weight,
            )
            pred = np.asarray(model.predict(X_inner_val), dtype=int)
            fold_metric = classification_metrics(item["y_val"], pred, labels=labels)
            fold_metric["PatientMacroF1"] = _patient_balanced_macro_f1(
                item["y_val"], pred,
                item.get("val_patient_groups", np.arange(item["y_val"].size)),
                labels,
            )
            fold_metric["InnerFold"] = int(item["fold"])
            fold_metric["ValidationSamples"] = int(item["y_val"].size)
            fold_metrics.append(fold_metric)
            if include_predictions:
                all_pred.append(pred)
                all_true.append(item["y_val"])
                try:
                    all_scores.append(
                        np.asarray(model._transform(X_inner_val) @ model.W_output, dtype=float)
                    )
                except Exception:
                    all_scores = []
            if record:
                self.actual_bls_fit_count += 1

        if record:
            # MAXFES remains the number of distinct candidate-mask evaluations.
            self.model_fit_count += 1

        metrics = _aggregate_inner_fold_metrics(
            fold_metrics,
            selected_ratio,
            macro_f1_std_penalty=self.macro_f1_std_penalty,
            patient_macro_weight=self.patient_macro_weight,
        )
        fitness = float(metrics["Fitness"])
        if self.mrmr_prior_enabled:
            selected_prior = float(np.mean(self.mrmr_prior_scores[selected_indices]))
            power = max(0.0, float(getattr(C, "MRMR_OBJECTIVE_POWER", 1.0)))
            prior_penalty = float(self.mrmr_objective_lambda) * (
                max(0.0, 1.0 - selected_prior) ** power
            )
            fitness += prior_penalty
            metrics["MRMRPriorCoverage"] = selected_prior
            metrics["MRMRPriorPenalty"] = prior_penalty
            metrics["MRMRObjectiveLambda"] = float(self.mrmr_objective_lambda)
        else:
            metrics["MRMRPriorCoverage"] = np.nan
            metrics["MRMRPriorPenalty"] = 0.0
            metrics["MRMRObjectiveLambda"] = 0.0
        metrics["BLSClassWeightMode"] = str(getattr(C, "BLS_CLASS_WEIGHT", "none"))
        if self.sample_weight is not None:
            metrics["BLSWeightMin"] = float(np.min(self.sample_weight))
            metrics["BLSWeightMax"] = float(np.max(self.sample_weight))
        metrics.update({
            "Fitness": float(fitness),
            "FinalBest": float(fitness),
            "SelectedCount": selected_count,
            "SelectedRatio": selected_ratio,
            "SelectedMask": mask.astype(int).tolist(),
            "SelectedFeatureNames": self.feature_names[mask].tolist(),
        })
        if include_predictions:
            metrics.update({
                "PredLabels": np.concatenate(all_pred).astype(int).tolist(),
                "TrueLabels": np.concatenate(all_true).astype(int).tolist(),
                "ScoreMatrix": (
                    np.vstack(all_scores).astype(float).tolist()
                    if len(all_scores) == len(self.validation_folds) else None
                ),
            })

        # cache 只保存轻量指标。不要把预测数组、raw score 存入 cache。
        if not include_predictions:
            self.cache[key] = (float(fitness), metrics)
            self.evaluated_masks[key] = mask.copy()
        elif key not in self.cache:
            light_metrics = dict(metrics)
            light_metrics.pop("PredLabels", None)
            light_metrics.pop("TrueLabels", None)
            light_metrics.pop("ScoreMatrix", None)
            self.cache[key] = (float(fitness), light_metrics)
            self.evaluated_masks[key] = mask.copy()

        if record:
            self._record_eval(float(fitness), metrics, mask, cached=False)
        return float(fitness), metrics

    def __call__(self, x: np.ndarray) -> float:
        # 优化器调用 objective 时只需要 fitness 标量，不保存预测数组，避免缓存膨胀。
        mask = self.vector_to_mask(x)
        f, _ = self.score_mask(mask, record=True, include_predictions=False)
        return float(f)

    def evaluate_vector(self, x: np.ndarray, record: bool = False, include_predictions: bool = True) -> Tuple[float, dict]:
        """返回某个向量对应的完整指标。

        默认 include_predictions=True，用于优化结束后保存最终 best_x 的预测结果；
        record=False 保证这个补全步骤不会额外写入 history，也不会增加正式预算。
        """
        mask = self.vector_to_mask(x)
        return self.score_mask(mask, record=record, include_predictions=include_predictions)


def prepare_fold_arrays(X, y, fold_data: FoldData):
    """在无数据泄漏前提下缩放训练、验证及可选测试数据。"""
    X_train_raw, X_val_raw = X[fold_data.train_idx], X[fold_data.val_idx]
    y_train, y_val = y[fold_data.train_idx], y[fold_data.val_idx]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)

    X_test = None
    y_test = None
    if fold_data.test_idx is not None and len(fold_data.test_idx) > 0:
        X_test_raw = X[fold_data.test_idx]
        y_test = y[fold_data.test_idx]
        # 仅用训练集拟合统计量，保持标准 train/validation/test 流程。
        X_test = scaler.transform(X_test_raw).astype(np.float32)
    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_inner_cv_arrays(X, y, fold_data: FoldData, patient_groups: Optional[np.ndarray] = None) -> list:
    """保存紧凑内折规格；每折缩放器只在其训练患者上拟合。"""
    raw_splits = tuple(getattr(fold_data, "inner_splits", ()) or ())
    if not raw_splits:
        return []
    prepared = []
    for inner_fold, (train_idx, val_idx) in enumerate(raw_splits, start=1):
        train_idx = np.asarray(train_idx, dtype=int)
        val_idx = np.asarray(val_idx, dtype=int)
        scaler = StandardScaler()
        scaler.fit(X[train_idx])
        prepared.append({
            "fold": int(inner_fold),
            "X_source": np.asarray(X, dtype=np.float32),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "y_train": np.asarray(y)[train_idx].astype(int),
            "y_val": np.asarray(y)[val_idx].astype(int),
            "scale_mean": np.asarray(scaler.mean_, dtype=np.float32),
            "scale_scale": np.asarray(scaler.scale_, dtype=np.float32),
            # 仅保存验证患者组 ID，供 ST_FODM_MRMR_PATIENT 计算患者等权指标；
            # 它不包含外层测试患者，也不包含任何外层测试标签。
            "val_patient_groups": (
                np.asarray(patient_groups)[val_idx].astype(str)
                if patient_groups is not None else np.asarray([], dtype=str)
            ),
        })
    return prepared


def _inner_fold_selected_arrays(item: dict, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """只物化当前 mask 的列，避免为每个 worker 常驻全部内折矩阵。"""
    mask = np.asarray(mask, dtype=bool)
    if "X_source" not in item:
        return item["X_train"][:, mask], item["X_val"][:, mask]
    selected = np.flatnonzero(mask)
    source = np.asarray(item["X_source"], dtype=np.float32)
    mean = np.asarray(item["scale_mean"], dtype=np.float32)[selected]
    scale = np.asarray(item["scale_scale"], dtype=np.float32)[selected]
    X_train = source[np.ix_(np.asarray(item["train_idx"], dtype=int), selected)]
    X_val = source[np.ix_(np.asarray(item["val_idx"], dtype=int), selected)]
    X_train = ((X_train - mean) / scale).astype(np.float32, copy=False)
    X_val = ((X_val - mean) / scale).astype(np.float32, copy=False)
    return X_train, X_val


def mrmr_objective_lambda_for_method(method: str) -> float:
    """Resolve formal-main lambda per mRMR-guided method.

    A process-level override is reserved for isolated lambda scans and bridge
    reproductions.  Otherwise the main experiment inherits the optimum from
    each method's own completed scan.
    """
    if getattr(C, "MRMR_OBJECTIVE_LAMBDA_OVERRIDE", None) is not None:
        return max(0.0, float(getattr(C, "MRMR_OBJECTIVE_LAMBDA", 0.0)))
    configured = getattr(C, "MRMR_OBJECTIVE_LAMBDA_BY_METHOD", {}) or {}
    if method in configured:
        return max(0.0, float(configured[method]))
    return max(0.0, float(getattr(C, "MRMR_OBJECTIVE_LAMBDA", 0.0)))


def _configured_final_label_ids(y_fit: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """在不查看留出测试标签的前提下确定最终标签空间。

AAMI 多分类固定编码为 0..N-1；其他调试模式由训练和验证标签定义。测试标签被显式
排除，保证最终预测头在查看测试集前已经完全冻结。"""
    task = str(
        getattr(C, "CLASSIFICATION_TASK", "multiclass") or "multiclass"
    ).lower()
    if task == "multiclass":
        observed = np.unique(np.concatenate([
            np.asarray(y_fit, dtype=int), np.asarray(y_val, dtype=int),
        ]))
        observed_width = int(np.max(observed)) + 1 if observed.size else 0
        n_classes = max(_effective_multiclass_num_classes(), observed_width)
        if n_classes >= 2:
            return np.arange(n_classes, dtype=int)

    labels = np.unique(
        np.concatenate([
            np.asarray(y_fit, dtype=int),
            np.asarray(y_val, dtype=int),
        ])
    )
    if labels.size < 2:
        raise ValueError("Final BLS evaluation requires at least two known classes.")
    return np.asarray(labels, dtype=int)


def evaluate_selected_mask_on_holdout_test(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_names,
    mask,
    seed: int,
    bls_seed: Optional[int] = None,
    test_patient_groups: Optional[np.ndarray] = None,
) -> dict:
    """使用 Plan 3 最终预测头及可选先验校正评价留出测试集。"""
    if X_test is None or y_test is None:
        return {}

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.size != np.asarray(X_train).shape[1] or int(mask.sum()) <= 0:
        raise ValueError("A non-empty selected mask matching the feature dimension is required for hold-out testing.")

    refit = bool(getattr(C, "HOLDOUT_REFIT_ON_TRAIN_VAL", False))
    if refit:
        X_fit = np.vstack([np.asarray(X_train), np.asarray(X_val)]).astype(np.float32)
        y_fit = np.concatenate([np.asarray(y_train), np.asarray(y_val)]).astype(int)
    else:
        X_fit = np.asarray(X_train, dtype=np.float32)
        y_fit = np.asarray(y_train, dtype=int)

    X_val_arr = np.asarray(X_val, dtype=np.float32)
    y_val_arr = np.asarray(y_val, dtype=int)
    X_test_arr = np.asarray(X_test, dtype=np.float32)
    y_test_arr = np.asarray(y_test, dtype=int)
    all_labels = _configured_final_label_ids(y_fit, y_val_arr)

    requested_size = max(1, int(getattr(C, "FINAL_BLS_ENSEMBLE_SIZE", 5)))
    offsets = list(getattr(C, "FINAL_BLS_ENSEMBLE_SEED_OFFSETS", [0, 97, 193, 389, 769]) or [0])
    if len(offsets) < requested_size:
        step = int(getattr(C, "FINAL_BLS_EXTRA_SEED_OFFSET_STEP", 7919))
        offsets = offsets + [step * (i + 1) for i in range(requested_size - len(offsets))]
    offsets = [int(v) for v in offsets[:requested_size]]

    base_seed = int(seed if bls_seed is None else bls_seed)
    temperature = max(
        float(getattr(C, "FINAL_BLS_ENSEMBLE_TEMPERATURE", 1.0)),
        float(getattr(C, "FINAL_BLS_TEMPERATURE_MIN", 1e-6)),
    )
    softmax_clip = float(getattr(C, "BLS_SOFTMAX_CLIP", 60.0))
    softmax_eps = float(getattr(C, "BLS_SOFTMAX_EPS", EPS))

    def _aligned_probs(model, X_arr: np.ndarray) -> np.ndarray:
        scores = model.decision_function(X_arr[:, mask]) / temperature
        scores = scores - np.max(scores, axis=1, keepdims=True)
        probs_local = np.exp(np.clip(scores, -softmax_clip, softmax_clip))
        probs_local = probs_local / np.maximum(probs_local.sum(axis=1, keepdims=True), softmax_eps)
        aligned = np.zeros((X_arr.shape[0], all_labels.size), dtype=np.float64)
        for col, cls in enumerate(np.asarray(model.classes_, dtype=int)):
            target = np.where(all_labels == cls)[0]
            if target.size:
                aligned[:, target[0]] += probs_local[:, col]
        return aligned

    prob_sum_test = np.zeros((X_test_arr.shape[0], all_labels.size), dtype=np.float64)
    prob_sum_val = np.zeros((X_val_arr.shape[0], all_labels.size), dtype=np.float64)
    successful = 0
    for offset in offsets:
        model = _make_bls_model(
            random_state=base_seed + int(offset),
            feature_indices=np.flatnonzero(mask),
            input_dim=int(mask.size),
        )
        model.fit(X_fit[:, mask], y_fit, sample_weight=_compute_bls_sample_weight(y_fit))
        prob_sum_test += _aligned_probs(model, X_test_arr)
        prob_sum_val += _aligned_probs(model, X_val_arr)
        successful += 1

    if successful <= 0:
        raise RuntimeError("No BLS ensemble member completed.")

    mean_test_probs = prob_sum_test / float(successful)
    mean_val_probs = prob_sum_val / float(successful)

    prior_enabled = bool(getattr(C, "FINAL_PRIOR_CORRECTION_ENABLED", False))
    prior_eps = float(getattr(C, "FINAL_PRIOR_CORRECTION_EPS", EPS))
    counts = np.asarray([np.sum(y_fit == lab) for lab in all_labels], dtype=np.float64)
    train_prior = (counts + prior_eps) / max(float(np.sum(counts) + prior_eps * len(counts)), prior_eps)

    def _corrected_scores(probs: np.ndarray, alpha: float) -> np.ndarray:
        return np.log(np.maximum(probs, prior_eps)) + float(alpha) * np.log(np.maximum(train_prior, prior_eps))

    def _metric_value(metrics_dict: dict, metric_name: str) -> float:
        key = str(metric_name or "accuracy").strip().lower().replace("_", "")
        mapping = {
            "accuracy": "Accuracy",
            "acc": "Accuracy",
            "weightedf1": "WeightedF1",
            "macrof1": "MacroF1",
            "balancedaccuracy": "BalancedAccuracy",
            "balacc": "BalancedAccuracy",
        }
        return float(metrics_dict.get(mapping.get(key, "Accuracy"), 0.0))

    best_alpha = 0.0
    best_prior_val_score = np.nan
    alpha_grid = [0.0]
    select_metric = str(getattr(C, "FINAL_PRIOR_CORRECTION_SELECT_METRIC", "accuracy"))
    if prior_enabled:
        raw_grid = getattr(C, "FINAL_PRIOR_CORRECTION_GRID", [0.0, 0.25, 0.5, 0.75, 1.0])
        alpha_grid = [float(v) for v in (raw_grid or [0.0])]
        best_tuple = (-np.inf, 0.0)
        for alpha in alpha_grid:
            val_pred = all_labels[np.argmax(_corrected_scores(mean_val_probs, alpha), axis=1)]
            val_metrics = classification_metrics(y_val_arr, np.asarray(val_pred, dtype=int), labels=all_labels.tolist())
            score = _metric_value(val_metrics, select_metric)
            candidate = (score, -abs(float(alpha)))
            if candidate > best_tuple:
                best_tuple = candidate
                best_alpha = float(alpha)
                best_prior_val_score = float(score)

    applied_alpha = best_alpha if prior_enabled else 0.0
    corrected_val_scores = _corrected_scores(mean_val_probs, applied_alpha)
    common_val_pred = all_labels[np.argmax(corrected_val_scores, axis=1)]
    common_val_metrics = classification_metrics(
        y_val_arr,
        np.asarray(common_val_pred, dtype=int),
        labels=all_labels.tolist(),
    )
    common_val_selected_ratio = float(mask.sum()) / max(1.0, float(mask.size))
    common_val_fitness = _fitness_from_metrics(
        common_val_metrics,
        common_val_selected_ratio,
    )

    corrected_test_scores = _corrected_scores(mean_test_probs, applied_alpha)
    pred = all_labels[np.argmax(corrected_test_scores, axis=1)]

    # 为保持旧 smoke test 和下游读取代码兼容，ScoreMatrixType 保持原名。
    # 是否做过先验校正由 FinalPriorCorrectionEnabled / Alpha 等字段记录。
    score_for_output = corrected_test_scores - np.max(corrected_test_scores, axis=1, keepdims=True)
    output_probs = np.exp(np.clip(score_for_output, -softmax_clip, softmax_clip))
    output_probs = output_probs / np.maximum(output_probs.sum(axis=1, keepdims=True), softmax_eps)

    metrics = classification_metrics(y_test_arr, np.asarray(pred, dtype=int), labels=all_labels.tolist())
    if test_patient_groups is not None:
        test_groups_arr = np.asarray(test_patient_groups).astype(str)
        if test_groups_arr.ndim != 1 or test_groups_arr.size != y_test_arr.size:
            raise ValueError("test_patient_groups must be a one-dimensional array aligned with y_test.")
        metrics["TestPatientMacroF1"] = _patient_balanced_macro_f1(
            y_test_arr,
            np.asarray(pred, dtype=int),
            test_groups_arr,
            all_labels.tolist(),
        )
        metrics["TestPatientCount"] = int(np.unique(test_groups_arr).size)
        metrics["TestPatientMacroF1Source"] = "outer_test_patient_equal_macro_f1"
    else:
        metrics["TestPatientMacroF1"] = np.nan
        metrics["TestPatientCount"] = 0
        metrics["TestPatientMacroF1Source"] = "unavailable_no_test_patient_groups"
    selected_ratio = float(mask.sum()) / max(1.0, float(mask.size))
    test_fitness = _fitness_from_metrics(metrics, selected_ratio)
    metrics["TestFitness"] = float(test_fitness)
    metrics["TestFitnessUtility"] = float(metrics.get("FitnessUtility", np.nan))
    actual_seeds = [base_seed + int(offset) for offset in offsets[:successful]]
    metrics.update({
        "PredLabels": np.asarray(pred, dtype=int).tolist(),
        "TrueLabels": y_test_arr.tolist(),
        "ScoreMatrix": output_probs.tolist(),
        "ScoreMatrixType": "ensemble_softmax_probability",
        "ScoreClassLabels": all_labels.astype(int).tolist(),
        "EvaluationSplit": "test",
        "SelectionSplit": "validation",
        "HoldoutRefitOnTrainVal": bool(refit),
        "FinalFitSource": "train+validation" if refit else "train",
        "FinalFitSamples": int(y_fit.size),
        "FinalPredictionHead": (
            "bls_seed_softmax_ensemble_trainval_refit"
            if refit else "bls_seed_softmax_ensemble"
        ),
        "FinalHeadFits": int(successful),
        "FinalHeadExtraFits": int(max(0, successful - 1)),
        "FinalEnsembleSize": int(successful),
        "FinalEnsembleTemperature": float(temperature),
        "FinalEnsembleSeedOffsets": [int(v) for v in offsets[:successful]],
        "FinalEnsembleSeeds": [int(v) for v in actual_seeds],
        "FinalPlanName": str(getattr(C, "FINAL_PLAN_NAME", "")),
        "FinalPriorCorrectionEnabled": bool(prior_enabled),
        "FinalPriorCorrectionAlpha": float(best_alpha if prior_enabled else 0.0),
        "FinalPriorCorrectionGrid": [float(v) for v in alpha_grid],
        "FinalPriorCorrectionSelectMetric": str(select_metric),
        "FinalPriorCorrectionValidationScore": float(best_prior_val_score) if prior_enabled else np.nan,
        "FinalTrainClassPrior": train_prior.tolist(),
        # 验证侧与测试侧使用同一套已拟合集成、温度、标签空间和先验 alpha，
        # 只有这种一致口径才适合用于
        # 计算同一预测头下的 validation-test gap。
        "CommonHeadValidationFitness": float(common_val_fitness),
        "CommonHeadValidationMacroF1": float(common_val_metrics.get("MacroF1", np.nan)),
        "CommonHeadValidationWeightedF1": float(common_val_metrics.get("WeightedF1", np.nan)),
        "CommonHeadValidationAccuracy": float(common_val_metrics.get("Accuracy", np.nan)),
        "CommonHeadValidationBalancedAccuracy": float(common_val_metrics.get("BalancedAccuracy", np.nan)),
        "CommonHeadValidationMinorityF1": float(common_val_metrics.get("MinorityF1", np.nan)),
        "CommonHeadValidationHardF1": float(common_val_metrics.get("HardF1", np.nan)),
        "CommonHeadValidationWorstMinorityF1": float(common_val_metrics.get("WorstMinorityF1", np.nan)),
        "CommonHeadValidationConfusionMatrix": common_val_metrics.get("ConfusionMatrix", []),
        "CommonHeadValidationClassMetrics": common_val_metrics.get("ClassMetrics", []),
        "CommonHeadValidationPredLabels": np.asarray(common_val_pred, dtype=int).tolist(),
        "CommonHeadValidationTrueLabels": y_val_arr.tolist(),
        "CommonHeadEvaluationNote": (
            "same final BLS ensemble/temperature/prior alpha on validation and test; "
            "HOLDOUT_REFIT_ON_TRAIN_VAL must remain False for leakage-free comparison"
        ),
    })
    return metrics

