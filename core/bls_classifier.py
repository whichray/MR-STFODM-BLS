# -*- coding: utf-8 -*-
"""实验五使用的轻量、确定性 BLS 分类器。

本版本额外保证了“特征选择过程中的 BLS 映射公平性”：
- 同一个 Fold × Run 内，所有候选 mask 共享同一套由原始特征索引决定的随机映射；
- 不同算法也可共享同一个 BLS 随机种子；
- 对不同特征数的 mask 做 sqrt(k) 输入缩放，避免大子集仅因随机投影方差增大而导致
  sigmoid/tanh 饱和。

这些机制不使用验证集或测试集标签；它们只使候选子集评价更可比、更可复现。
"""
from __future__ import annotations

from typing import Iterable, Optional
import numpy as np


class BroadLearningSystem:
    """轻量化 Broad Learning System（BLS）分类器。

    Parameters
    ----------
    feature_indices:
        当前输入列对应的“原始全局特征索引”。当 ``feature_mapping_mode`` 为
        ``"stable_global"`` 时，模型先根据 ``input_dim`` 生成完整特征空间的随机
        权重，再按此索引取行。这样同一个原始特征在不同 mask 中始终使用同一随机映射。
    input_dim:
        原始、未选特征空间的维度。若未提供，则退化为当前输入维度，兼容旧用法。
    feature_mapping_mode:
        ``"stable_global"``（推荐）或 ``"legacy_selected_order"``（旧版复现实验）。
    input_scale_mode:
        ``"sqrt_selected"``（推荐）或 ``"none"``。前者按 ``sqrt(已选特征数)``
        缩放映射层输入，控制不同 mask 宽度下的激活方差。
    activation_clip / sample_weight_eps / softmax_clip / softmax_eps:
        BLS 内部数值稳定参数，主实验由 ``config.py`` 统一传入。
    """

    def __init__(
        self,
        N1=100,
        N2=1,
        N3=500,
        C=2 ** -30,
        random_state=2026,
        *,
        feature_indices: Optional[Iterable[int]] = None,
        input_dim: Optional[int] = None,
        feature_mapping_mode: str = "stable_global",
        input_scale_mode: str = "sqrt_selected",
        activation_clip: float = 60.0,
        sample_weight_eps: float = 1e-12,
        softmax_clip: float = 60.0,
        softmax_eps: float = 1e-12,
    ):
        self.N1 = int(N1)
        self.N2 = int(N2)
        self.N3 = int(N3)
        self.C = float(C)
        self.random_state = None if random_state is None else int(random_state)

        self.feature_indices = None if feature_indices is None else np.asarray(feature_indices, dtype=int).reshape(-1)
        self.input_dim = None if input_dim is None else int(input_dim)
        self.feature_mapping_mode = str(feature_mapping_mode).strip().lower()
        self.input_scale_mode = str(input_scale_mode).strip().lower()
        # 以下数值稳定参数由 config.py 传入；保留默认值是为了该类可单独导入测试。
        self.activation_clip = abs(float(activation_clip))
        self.sample_weight_eps = max(float(sample_weight_eps), 1e-300)
        self.softmax_clip = abs(float(softmax_clip))
        self.softmax_eps = max(float(softmax_eps), 1e-300)
        if self.feature_mapping_mode not in {"stable_global", "legacy_selected_order"}:
            raise ValueError("feature_mapping_mode must be 'stable_global' or 'legacy_selected_order'.")
        if self.input_scale_mode not in {"sqrt_selected", "none"}:
            raise ValueError("input_scale_mode must be 'sqrt_selected' or 'none'.")

        self.W_feature = []
        self.W_enhancement = None
        self.b_enhancement = None
        self.W_output = None
        self.classes_ = None
        self._rng = np.random.default_rng(self.random_state)
        self._input_scale = 1.0
        self.n_input_features_ = None

    def _generate_weights(self, n_features: int):
        """为当前特征掩码生成可复现的 BLS 随机映射权重。"""
        n_features = int(n_features)
        if n_features <= 0:
            raise ValueError("BLS requires at least one input feature.")

        self.W_feature = []
        self.n_input_features_ = n_features
        self._input_scale = 1.0 / np.sqrt(float(n_features)) if self.input_scale_mode == "sqrt_selected" else 1.0

        if self.feature_mapping_mode == "stable_global" and self.feature_indices is not None:
            if self.feature_indices.size != n_features:
                raise ValueError("feature_indices must have exactly one index per selected input column.")
            total_dim = self.input_dim if self.input_dim is not None else int(np.max(self.feature_indices)) + 1
            if total_dim <= int(np.max(self.feature_indices)):
                raise ValueError("input_dim must exceed every feature_indices entry.")
            if np.any(self.feature_indices < 0):
                raise ValueError("feature_indices must be non-negative.")

            for _ in range(self.N2):
                full_w = self._rng.standard_normal((total_dim, self.N1))
                self.W_feature.append(full_w[self.feature_indices, :])
        else:
            # 兼容模式：复现历史版本按已选列顺序重新生成映射的行为。
            for _ in range(self.N2):
                self.W_feature.append(self._rng.standard_normal((n_features, self.N1)))

        # 随 N1×N2 增大时限制增强节点输入，保证 tanh 数值稳定。
        feature_node_dim = self.N1 * self.N2
        self.W_enhancement = self._rng.standard_normal((feature_node_dim, self.N3)) / np.sqrt(float(feature_node_dim))
        self.b_enhancement = self._rng.standard_normal((1, self.N3))

    def _sigmoid(self, A):
        A = np.clip(A, -self.activation_clip, self.activation_clip)
        return 1.0 / (1.0 + np.exp(-A))

    def _transform(self, X):
        if self.W_enhancement is None or self.b_enhancement is None:
            raise RuntimeError("BLS has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if self.n_input_features_ is not None and X.shape[1] != self.n_input_features_:
            raise ValueError(
                f"X has {X.shape[1]} columns but this BLS was fitted with "
                f"{self.n_input_features_} selected features."
            )

        Z_list = []
        for W in self.W_feature:
            Z_list.append(self._sigmoid((X @ W) * self._input_scale))
        Z_stack = np.hstack(Z_list)
        H = (Z_stack @ self.W_enhancement) + self.b_enhancement
        E = np.tanh(H)
        return np.hstack([Z_stack, E])

    def fit(self, X, y, sample_weight=None):
        """使用可选样本权重的岭回归拟合 BLS 输出层。"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        if sample_weight is None:
            sqrt_weight = None
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError("sample_weight must have one value per training sample.")
            if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight <= 0):
                raise ValueError("sample_weight must contain finite positive values.")
            sqrt_weight = np.sqrt(np.maximum(sample_weight, self.sample_weight_eps))

        self.classes_ = np.unique(y)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[v] for v in y], dtype=int)
        Y = np.eye(len(self.classes_), dtype=np.float64)[y_idx]

        self._rng = np.random.default_rng(self.random_state)
        self._generate_weights(X.shape[1])
        P = self._transform(X)

        if sqrt_weight is None:
            Pw, Yw = P, Y
        else:
            Pw = P * sqrt_weight[:, None]
            Yw = Y * sqrt_weight[:, None]

        A = Pw.T @ Pw + self.C * np.eye(P.shape[1], dtype=np.float64)
        B = Pw.T @ Yw
        try:
            self.W_output = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            self.W_output = np.linalg.lstsq(A, B, rcond=None)[0]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        P = self._transform(X)
        scores = P @ self.W_output
        return self.classes_[np.argmax(scores, axis=1)]

    def decision_function(self, X):
        """按 classes_ 的固定类别顺序返回 BLS 原始类别分数。

公开该接口可以避免下游代码直接访问私有 _transform，并用于验证集校准及
概率级多随机种子集成。"""
        if self.W_output is None or self.classes_ is None:
            raise RuntimeError("BLS has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        return self._transform(X) @ self.W_output

    def predict_proba(self, X):
        """对 BLS 原始分数执行数值稳定的 softmax。

BLS 输出层由岭回归训练而非概率损失，因此这里得到的是分数归一化后的伪概率。
它只用于显式启用的最终阶段流程；普通 predict 仍保持历史上的原始分数 argmax 行为。"""
        scores = self.decision_function(X)
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(shifted, -self.softmax_clip, self.softmax_clip))
        return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), self.softmax_eps)

