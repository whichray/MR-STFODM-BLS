# -*- coding: utf-8 -*-
"""
实验五鲁棒文件保存工具。

设计目标：
    1. 尽量避免实验中途崩溃导致结果文件损坏；
    2. 每个 Fold/Run/Method 任务单独落盘，方便断点续跑；
    3. 保存足够的环境、数据和任务元信息，保证后续可追溯、可复算、可画图；
    4. 尽量少依赖第三方库，保证在多进程 joblib/loky 环境下稳定工作。

说明：
    本文件只提供 IO 辅助函数，不包含任何实验算法逻辑。
"""
from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def now_iso() -> str:
    """返回当前时间的 ISO 字符串，用于日志和 task_status。"""
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path | str) -> Path:
    """确保目录存在，并返回 Path 对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_default(obj: Any):
    """
    json.dump 的兜底转换函数。

    numpy / pandas 类型无法直接写入 JSON，因此这里统一转成 Python 原生类型。
    NaN / Inf 会转成 None，避免生成非法 JSON。
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    try:
        import pandas as pd  # 局部导入，避免工具函数对 pandas 版本过度敏感
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def atomic_write_text(text: str, path: Path | str, encoding: str = "utf-8") -> None:
    """
    原子写入文本文件。

    写入流程：
        1. 先在目标目录写入临时文件；
        2. 写完后用 os.replace 一次性替换目标文件。

    好处：
        如果程序在写文件中途崩溃，不会留下半个损坏的正式文件。
    """
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def atomic_write_json(obj: Any, path: Path | str, indent: int = 2) -> None:
    """原子写入 JSON 文件，默认保留中文。"""
    text = json.dumps(obj, ensure_ascii=False, indent=indent, default=json_default)
    atomic_write_text(text, path, encoding="utf-8")


def atomic_write_dataframe(df: pd.DataFrame, path: Path | str, encoding: str = "utf-8-sig") -> None:
    """
    原子写入 CSV 文件。

    默认使用 utf-8-sig，Windows Excel 直接打开中文列名时不容易乱码。
    """
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp_name, index=False, encoding=encoding)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def safe_save_npz(path: Path | str, **arrays: Any) -> None:
    """安全保存压缩 npz 文件，常用于预测结果 y_true/y_pred/y_score。"""
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp.npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def safe_save_npy(path: Path | str, array: Any) -> None:
    """安全保存 npy 文件，主要用于每个任务的最终 selected mask。"""
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp.npy", dir=str(path.parent))
    os.close(fd)
    try:
        np.save(tmp_name, np.asarray(array))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def read_json(path: Path | str) -> Any:
    """读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path: Path | str, block_size: int = 1 << 20) -> str:
    """
    计算文件 SHA256。

    用途：
        保存到 dataset_info.json，后续确认不同实验是否使用同一个数据文件。
    """
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def environment_snapshot() -> dict:
    """
    获取当前 Python 环境信息。

    用途：
        写入 environment.json，便于后续复现实验。
    """
    out = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    for mod_name in ["numpy", "pandas", "sklearn", "scipy", "joblib", "threadpoolctl"]:
        try:
            mod = __import__(mod_name)
            out[f"{mod_name}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[f"{mod_name}_version"] = None
    return out


def task_id(method: str, fold: int, run: int) -> str:
    """
    生成统一任务编号。

    例如：
        method='ST_FODM', fold=1, run=2 -> F01_R02_ST_FODM

    该编号会用于 partial/status/history/mask/prediction 等文件名，
    因此不要随意修改格式，否则会影响断点续跑。
    """
    safe_method = str(method).replace("/", "-").replace(" ", "_")
    return f"F{int(fold):02d}_R{int(run):02d}_{safe_method}"


def traceback_text() -> str:
    """返回当前异常的完整 traceback 字符串，用于 errors.log 和 failed task JSON。"""
    return traceback.format_exc()
