from __future__ import annotations
from collections import deque
import numpy as np

class EvaluationBudgetExceeded(RuntimeError):
    pass


class FO_Direction_CEO_STAGNATION_REFINE:
    """带停滞触发局部精修的 FODM。

该版本不再固定划分全局/局部预算，而是在完成最低比例的 FODM 搜索后，
根据连续未改进的真实评价次数触发局部精修；若始终未触发，则在预算末段强制
预留少量评价用于精修。总 MaxFES 保持不变。"""

    def __init__(
        self,
        func,
        Dim,
        lb,
        ub,
        Np=50,
        N=5,
        MaxFES=1000,
        schedule_max_fes=None,
        alpha=0.6,
        M=4,
        lambda_chaos=0.7,
        min_fodm_ratio=0.5,
        stagnation_trigger_patience=100,
        force_refine_ratio=0.1,
        local_mode="swap_or_flip",
        local_patience=999999,
        stagnation_tol=1e-12,
        initial_seed_masks=None,
        **kwargs,
    ):
        self.func = func
        self.Dim = int(Dim)
        self.lb = np.asarray(lb, dtype=float).reshape(-1)
        self.ub = np.asarray(ub, dtype=float).reshape(-1)
        self.Np = int(Np) + int(Np % 2 == 1)
        self.N = int(N)
        self.MaxFES = int(MaxFES)
        # MaxFES 是搜索阶段的硬停止预算；schedule_max_fes 只控制按比例计算的
        # 阶段切换点和自适应局部搜索日程。普通方法二者相同；A2/A2-E 会预留
        # 固定复评预算，并用剩余预算作为搜索硬上限，同时保留完整 MAXFES 的
        # BR_A 调度比例，从而形成干净的预算重分配消融。
        self.schedule_max_fes = max(
            self.MaxFES,
            int(self.MaxFES if schedule_max_fes is None else schedule_max_fes),
        )

        self.alpha = float(alpha)
        self.M = int(M)
        self.lambda_chaos = float(lambda_chaos)
        if not (0.0 <= self.lambda_chaos <= 1.0):
            raise ValueError("lambda_chaos must be in [0, 1].")

        self.min_fodm_ratio = min(0.95, max(0.0, float(min_fodm_ratio)))
        self.stagnation_trigger_patience = max(1, int(stagnation_trigger_patience))
        self.force_refine_ratio = min(0.9, max(0.0, float(force_refine_ratio)))
        self.local_mode = str(local_mode)
        self.local_patience = max(1, int(local_patience))
        self.stagnation_tol = float(stagnation_tol)
        # Optional training-only seeds used by Experiment 37's mRMR-guided
        # Base ST-FODM.  With no seeds, the original ST-FODM path is unchanged.
        self.initial_seed_masks = [
            np.asarray(mask, dtype=bool).reshape(-1).copy()
            for mask in (initial_seed_masks or [])
            if np.asarray(mask, dtype=bool).size == self.Dim
        ]

        self.FEvals = 0
        self.RawEvalCalls = 0
        self.Population = None
        self.fitness = None
        self.Best = None
        self.fBest = np.inf
        self.direction_hist = None
        self.w = self.compute_fractional_weights()
        self.iter = 0
        self.history = []
        self.fe_history = []

        # 以下诊断字段由主程序写入运行明细和汇总表。
        self.switch_trigger_reason = ""
        self.switch_fevals = np.nan
        self.fodm_fevals = 0
        self.last_global_improve_fe = np.nan
        self.no_improve_fe_before_switch = np.nan
        self.local_refine_fevals = 0
        self.local_refine_improvements = 0
        self.local_refine_start_fitness = np.nan
        self.local_refine_end_fitness = np.nan
        self.local_refine_start_vector = None
        self.local_refine_end_vector = None

        # 预算补足仅在正常全局/局部流程提前停止后启用，不改变 BR_A/A2
        # 的引导、接受、成功记忆或稳健复评规则。它的唯一职责是生成尚未
        # 评价的 mask，把搜索阶段真实 ModelFits 补足到 MaxFES。
        self.budget_fill_model_fits = 0
        self.budget_fill_attempts = 0
        self.budget_fill_cache_avoided = 0
        self.budget_fill_duplicate_avoided = 0
        self.budget_fill_random_restarts = 0
        self.budget_fill_improvements = 0
        self.no_novel_candidate_count = 0
        self.final_unused_budget = int(self.MaxFES)
        self._budget_fill_seen_keys = set()

    def _evaluate(self, x):
        if self.FEvals >= self.MaxFES:
            raise EvaluationBudgetExceeded()
        raw_limit = int(getattr(self.func, "raw_eval_call_limit", max(self.MaxFES, self.MaxFES * 8)))
        if self.RawEvalCalls >= raw_limit:
            raise EvaluationBudgetExceeded()
        before = getattr(self.func, "model_fit_count", None)
        value = float(self.func(np.asarray(x, dtype=float).reshape(-1)))
        self.RawEvalCalls += 1
        after = getattr(self.func, "model_fit_count", None)
        effective_mode = str(getattr(self.func, "budget_count_mode", "raw_calls")) == "effective_model_fits"
        if effective_mode and before is not None and after is not None:
            self.FEvals += max(0, int(after) - int(before))
        else:
            self.FEvals += 1
        return value

    def compute_fractional_weights(self):
        w = np.zeros(self.M + 1, dtype=float)
        w[0] = 1.0
        for j in range(1, self.M + 1):
            w[j] = (1.0 - (self.alpha + 1.0) / j) * w[j - 1]
        return w

    def EDM(self, x, y):
        k = 2.66
        return k * (np.exp(-np.cos(np.pi * y)) - 1.0) * x, y + x

    def bound_constraint(self, v):
        v = np.asarray(v, dtype=float).copy()
        low_state = np.tile(self.lb, (v.shape[0], 1))
        up_state = np.tile(self.ub, (v.shape[0], 1))
        v[v < low_state] = np.minimum(up_state[v < low_state], 2 * low_state[v < low_state] - v[v < low_state])
        v[v > up_state] = np.maximum(low_state[v > up_state], 2 * up_state[v > up_state] - v[v > up_state])
        return v

    def initialization(self):
        self.Population = self.lb + np.random.rand(self.Np, self.Dim) * (self.ub - self.lb)
        # Encode selected features above the transfer threshold and keep
        # unselected features random below it.  The perturbation avoids a
        # collapsed, identical initial population.
        for idx, mask in enumerate(self.initial_seed_masks[:self.Np]):
            x = self.lb + np.random.rand(self.Dim) * 0.30 * (self.ub - self.lb)
            selected = np.flatnonzero(mask)
            if selected.size:
                x[selected] = self.lb[selected] + (
                    0.70 + 0.30 * np.random.rand(selected.size)
                ) * (self.ub[selected] - self.lb[selected])
            self.Population[idx] = self._clip_vector(x)
        self.fitness = np.array([self._evaluate(self.Population[i]) for i in range(self.Np)], dtype=float)
        best_idx = int(np.argmin(self.fitness))
        self.Best = self.Population[best_idx].copy()
        self.fBest = float(self.fitness[best_idx])
        self.history = [self.fBest]
        self.fe_history = [self.FEvals]
        self.iter = 0
        self.direction_hist = [deque(maxlen=self.M + 1) for _ in range(self.Np)]
        self.last_global_improve_fe = int(self.FEvals)

    def fractional_direction_memory(self, idx, current_direction=None):
        hist = self.direction_hist[int(idx)]
        if len(hist) == 0:
            if current_direction is None:
                return np.zeros(self.Dim, dtype=float)
            return np.asarray(current_direction, dtype=float).copy()

        g = np.zeros(self.Dim, dtype=float)
        length = min(len(hist), self.M + 1)
        for j in range(length):
            g += self.w[j] * hist[-1 - j]
        return g

    def update_direction_history(self, idx, direction):
        self.direction_hist[int(idx)].append(np.asarray(direction, dtype=float).copy())

    def _clip_vector(self, x):
        return np.minimum(self.ub, np.maximum(self.lb, np.asarray(x, dtype=float).reshape(-1)))

    def _vector_to_mask(self, x):
        if hasattr(self.func, "vector_to_mask"):
            return np.asarray(self.func.vector_to_mask(x), dtype=bool)
        threshold = float(getattr(self.func, "threshold", 0.5))
        mask = np.asarray(x, dtype=float).reshape(-1) > threshold
        min_features = int(getattr(self.func, "min_features", 1))
        if int(mask.sum()) < min_features:
            order = np.argsort(np.asarray(x, dtype=float).reshape(-1))[::-1]
            mask[:] = False
            mask[order[:min_features]] = True
        return mask

    def _mask_to_vector(self, mask):
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        x = np.where(mask, self.ub, self.lb).astype(float)
        return self._clip_vector(x)

    def _mask_key_bytes(self, mask):
        """返回与 objective 缓存兼容的二进制 mask 键。"""
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        try:
            if hasattr(self.func, "mask_key"):
                return self.func.mask_key(mask)
        except Exception:
            pass
        return np.packbits(mask.astype(np.uint8)).tobytes()

    def _is_cached_mask(self, mask):
        """检查 mask 是否已经产生过真实模型评价。"""
        try:
            key = self._mask_key_bytes(mask)
            return hasattr(self.func, "cache") and key in self.func.cache
        except Exception:
            return False

    def _repair_min_features(self, mask):
        """修复随机候选，确保满足最小特征数约束。"""
        mask = np.asarray(mask, dtype=bool).reshape(-1).copy()
        min_features = max(1, int(getattr(self.func, "min_features", 1)))
        if int(mask.sum()) < min_features:
            missing = min_features - int(mask.sum())
            available = np.flatnonzero(~mask)
            if available.size > 0:
                chosen = np.random.choice(
                    available,
                    size=min(missing, int(available.size)),
                    replace=False,
                )
                mask[np.asarray(chosen, dtype=int)] = True
        return mask

    def _novel_budget_fill_mask(self, reference_mask):
        """生成一个未缓存的预算补足候选。

        前半部分在当前最优 mask 周围执行逐渐扩大的多 bit 扰动，尽量保留
        算法已找到的有效结构；后半部分使用随机重启，避免局部邻域被缓存
        完全覆盖。该函数只在正常搜索提前停止后调用。
        """
        reference = self._repair_min_features(reference_mask)
        max_attempts = max(128, int(self.Dim) * 6)
        reference_count = int(reference.sum())
        min_features = max(1, int(getattr(self.func, "min_features", 1)))

        for attempt in range(max_attempts):
            self.budget_fill_attempts += 1
            if attempt < max_attempts // 2:
                candidate = reference.copy()
                max_flips = max(2, min(self.Dim, 1 + attempt // 8))
                n_flips = int(np.random.randint(1, max_flips + 1))
                flip_idx = np.random.choice(
                    self.Dim,
                    size=min(n_flips, self.Dim),
                    replace=False,
                )
                candidate[np.asarray(flip_idx, dtype=int)] = ~candidate[
                    np.asarray(flip_idx, dtype=int)
                ]
            else:
                # 随机重启的特征数围绕当前最优规模波动，同时允许更远区域。
                self.budget_fill_random_restarts += 1
                if np.random.rand() < 0.70:
                    radius = max(2, int(round(0.20 * self.Dim)))
                    low = max(min_features, reference_count - radius)
                    high = min(self.Dim, reference_count + radius)
                    k = int(np.random.randint(low, high + 1))
                else:
                    k = int(np.random.randint(min_features, self.Dim + 1))
                candidate = np.zeros(self.Dim, dtype=bool)
                chosen = np.random.choice(self.Dim, size=k, replace=False)
                candidate[np.asarray(chosen, dtype=int)] = True

            candidate = self._repair_min_features(candidate)
            key = self._mask_key_bytes(candidate)
            if key in self._budget_fill_seen_keys:
                self.budget_fill_duplicate_avoided += 1
                continue
            if self._is_cached_mask(candidate):
                self.budget_fill_cache_avoided += 1
                continue
            self._budget_fill_seen_keys.add(key)
            return candidate

        self.no_novel_candidate_count += 1
        return None

    def _run_budget_fill(self):
        """用未评价候选补足剩余真实 ModelFit 预算。

        正常算法阶段可能因为局部耐心提前结束，或因重复/cache mask 触发
        RawEvalCalls 安全上限。补足阶段先扩展原始调用安全额度，然后只评价
        已确认未缓存的 mask，因此不会把缓存命中错误计为真实预算。
        """
        if self.FEvals >= self.MaxFES or self.Best is None:
            self.final_unused_budget = max(0, int(self.MaxFES - self.FEvals))
            return

        remaining = max(0, int(self.MaxFES - self.FEvals))
        try:
            current_limit = int(
                getattr(
                    self.func,
                    "raw_eval_call_limit",
                    max(self.MaxFES, self.MaxFES * 8),
                )
            )
            self.func.raw_eval_call_limit = max(
                current_limit,
                int(self.RawEvalCalls) + remaining + 64,
            )
        except Exception:
            pass

        reference_mask = self._vector_to_mask(self.Best)
        while self.FEvals < self.MaxFES:
            candidate_mask = self._novel_budget_fill_mask(reference_mask)
            if candidate_mask is None:
                break

            candidate_x = self._mask_to_vector(candidate_mask)
            before_fe = int(self.FEvals)
            candidate_f = self._evaluate_vector_counted(candidate_x)
            consumed = max(0, int(self.FEvals) - before_fe)
            if consumed <= 0:
                # 理论上候选已通过缓存检查；这里保留竞态/键实现差异的兜底。
                self.budget_fill_cache_avoided += 1
                continue

            self.budget_fill_model_fits += consumed
            if candidate_f < self.fBest - self.stagnation_tol:
                self.fBest = float(candidate_f)
                self.Best = candidate_x.copy()
                reference_mask = candidate_mask.copy()
                self.budget_fill_improvements += 1

            self.history.append(float(self.fBest))
            self.fe_history.append(int(self.FEvals))

        self.final_unused_budget = max(0, int(self.MaxFES - self.FEvals))

    def _evaluate_vector_counted(self, x):
        if self.FEvals >= self.MaxFES:
            raise EvaluationBudgetExceeded()
        raw_limit = int(getattr(self.func, "raw_eval_call_limit", max(self.MaxFES, self.MaxFES * 8)))
        if self.RawEvalCalls >= raw_limit:
            raise EvaluationBudgetExceeded()
        before = getattr(self.func, "model_fit_count", None)
        value = float(self.func(self._clip_vector(x)))
        self.RawEvalCalls += 1
        after = getattr(self.func, "model_fit_count", None)
        effective_mode = str(getattr(self.func, "budget_count_mode", "raw_calls")) == "effective_model_fits"
        if effective_mode and before is not None and after is not None:
            self.FEvals += max(0, int(after) - int(before))
        else:
            self.FEvals += 1
        return value

    def _candidate_mask(self, mask):
        mask = np.asarray(mask, dtype=bool).copy()
        selected = np.where(mask)[0]
        unselected = np.where(~mask)[0]
        min_features = int(getattr(self.func, "min_features", 1))

        if self.local_mode == "swap_or_flip" and selected.size > 0 and unselected.size > 0 and np.random.rand() < 0.5:
            # swap 保持特征数量不变，用于测试“替换特征”而不是简单增删。
            out_idx = int(np.random.choice(selected))
            in_idx = int(np.random.choice(unselected))
            mask[out_idx] = False
            mask[in_idx] = True
            return mask

        # 单 bit 翻转；若删除会违反最小特征数约束，则改为添加。
        j = int(np.random.randint(0, self.Dim))
        if mask[j] and int(mask.sum()) <= min_features:
            if unselected.size > 0:
                j = int(np.random.choice(unselected))
                mask[j] = True
            return mask

        mask[j] = ~mask[j]
        if int(mask.sum()) < min_features:
            order = np.random.permutation(self.Dim)
            for k in order:
                mask[int(k)] = True
                if int(mask.sum()) >= min_features:
                    break
        return mask

    def _should_switch(self, min_global_fevals, force_refine_start):
        if self.FEvals >= force_refine_start:
            return True, "force_refine_budget"
        if self.FEvals >= min_global_fevals:
            no_improve_fe = self.FEvals - int(self.last_global_improve_fe)
            if no_improve_fe >= self.stagnation_trigger_patience:
                return True, "stagnation_trigger"
        return False, ""

    def _run_fodm_until_switch(self):
        min_global_fevals = int(round(self.schedule_max_fes * self.min_fodm_ratio))
        force_refine_start = int(round(self.schedule_max_fes * (1.0 - self.force_refine_ratio)))

        # 安全约束：预算允许时，至少为局部精修保留一次真实评价。
        force_refine_start = max(self.Np + 1, min(self.MaxFES - 1, force_refine_start))
        min_global_fevals = max(self.Np + 1, min(force_refine_start, min_global_fevals))

        self.initialization()

        stop = False
        reason = ""
        while self.FEvals < self.MaxFES and not stop:
            should, reason = self._should_switch(min_global_fevals, force_refine_start)
            if should:
                break

            rand_index = np.random.permutation(self.Np)
            ubp = np.max(self.Population, axis=0)
            lbp = np.min(self.Population, axis=0)

            for i in range(0, self.Np, 2):
                idx = rand_index[i:i + 2]
                xy = self.Population[idx]

                xy_dot = (xy - lbp) / ((ubp - lbp) + np.finfo(float).eps) * (
                    np.tile(np.array([0.5, 0.25]), (self.Dim, 1)).T -
                    np.tile(np.array([-0.5, -0.25]), (self.Dim, 1)).T
                ) + np.tile(np.array([-0.5, -0.25]), (self.Dim, 1)).T

                x = xy_dot[0].copy()
                y = xy_dot[1].copy()
                x_chaos, y_chaos = [], []
                for _ in range(self.N):
                    x, y = self.EDM(x, y)
                    x_chaos.append(x)
                    y_chaos.append(y)
                xy_chaos = [np.asarray(x_chaos, dtype=float), np.asarray(y_chaos, dtype=float)]

                for k in range(2):
                    xy_chaos_dot = ((xy_chaos[k] - np.array([-0.5, -0.25])[k]) /
                                    (np.array([0.5, 0.25])[k] - np.array([-0.5, -0.25])[k])) * (ubp - lbp) + lbp
                    xy_chaos_dot = self.bound_constraint(xy_chaos_dot)

                    d_chaos = xy_chaos_dot - xy[k]
                    d_rep = np.mean(d_chaos, axis=0)

                    cur_idx = int(idx[k])
                    g = self.fractional_direction_memory(cur_idx, current_direction=d_rep)
                    d_bar = self.lambda_chaos * d_chaos + (1.0 - self.lambda_chaos) * g

                    a = np.random.rand(self.N, 1)
                    if np.random.rand() < 0.5:
                        xy_hat = xy[k] + a * d_bar
                    else:
                        xy_hat = self.Best + a * d_bar

                    cr = np.random.rand()
                    trials = []
                    for n in range(self.N):
                        trial = xy[k].copy()
                        jrand = np.random.randint(self.Dim)
                        for j in range(self.Dim):
                            if np.random.rand() <= cr or j == jrand:
                                trial[j] = xy_hat[n, j]
                        trials.append(trial)

                    trials = np.asarray(trials, dtype=float)
                    trials = self.bound_constraint(trials)

                    f_trials = []
                    for trial in trials:
                        f_trials.append(self._evaluate(trial))
                    f_trials = np.asarray(f_trials, dtype=float)

                    best_trial_idx = int(np.argmin(f_trials))
                    if f_trials[best_trial_idx] <= self.fitness[cur_idx]:
                        self.Population[cur_idx] = trials[best_trial_idx]
                        self.fitness[cur_idx] = f_trials[best_trial_idx]
                        if f_trials[best_trial_idx] < self.fBest - self.stagnation_tol:
                            self.Best = trials[best_trial_idx].copy()
                            self.fBest = float(f_trials[best_trial_idx])
                            self.last_global_improve_fe = int(self.FEvals)

                    self.update_direction_history(cur_idx, d_rep)

                    should, reason = self._should_switch(min_global_fevals, force_refine_start)
                    if should or self.FEvals >= self.MaxFES:
                        stop = True
                        break

                if stop:
                    break

            self.history.append(float(self.fBest))
            self.fe_history.append(int(self.FEvals))
            self.iter += 1
            if self.iter % 10 == 0:
                print(f"iter={self.iter}  FE={self.FEvals}  ObjVal={self.fBest:.16f}")

        if not reason:
            if self.FEvals >= self.MaxFES:
                reason = "maxfes_exhausted"
            else:
                reason = "unknown"

        self.switch_trigger_reason = reason
        self.switch_fevals = int(self.FEvals)
        self.fodm_fevals = int(self.FEvals)
        self.no_improve_fe_before_switch = int(self.FEvals - int(self.last_global_improve_fe))
        return reason

    def _run_local_refinement(self):
        if self.FEvals >= self.MaxFES:
            self.local_refine_start_fitness = float(self.fBest)
            self.local_refine_end_fitness = float(self.fBest)
            self.local_refine_start_vector = None if self.Best is None else np.asarray(self.Best, dtype=float).copy()
            self.local_refine_end_vector = None if self.Best is None else np.asarray(self.Best, dtype=float).copy()
            return

        current_mask = self._vector_to_mask(self.Best)
        current_x = self._mask_to_vector(current_mask)
        current_f = float(self.fBest)

        self.local_refine_start_fitness = float(current_f)
        self.local_refine_start_vector = current_x.copy()
        no_improve = 0

        while self.FEvals < self.MaxFES:
            candidate_mask = self._candidate_mask(current_mask)
            candidate_x = self._mask_to_vector(candidate_mask)
            before_fe = int(self.FEvals)
            candidate_f = self._evaluate_vector_counted(candidate_x)
            self.local_refine_fevals += max(0, int(self.FEvals) - before_fe)

            improved = candidate_f < current_f - self.stagnation_tol
            accepted = candidate_f <= current_f

            if accepted:
                current_mask = candidate_mask
                current_x = candidate_x
                current_f = float(candidate_f)

            if improved:
                no_improve = 0
                self.local_refine_improvements += 1
                if candidate_f < self.fBest - self.stagnation_tol:
                    self.fBest = float(candidate_f)
                    self.Best = candidate_x.copy()
            else:
                no_improve += 1

            self.history.append(float(self.fBest))
            self.fe_history.append(int(self.FEvals))

            if no_improve >= self.local_patience:
                break

        self.local_refine_end_fitness = float(self.fBest)
        self.local_refine_end_vector = None if self.Best is None else np.asarray(self.Best, dtype=float).copy()

    def optimize(self):
        try:
            self._run_fodm_until_switch()
            self._run_local_refinement()
        except EvaluationBudgetExceeded:
            pass

        # 无论正常流程因局部耐心还是 RawEvalCalls 安全阈值提前停止，都使用
        # 未缓存候选补齐搜索预算。A2/A2-E 的 MaxFES 在此处仅指搜索预算，
        # 后续稳健复评预算仍由主程序单独计入，总预算保持不变。
        try:
            self._run_budget_fill()
        except EvaluationBudgetExceeded:
            self.final_unused_budget = max(0, int(self.MaxFES - self.FEvals))

        if len(self.history) == 0:
            self.history = [float(self.fBest)]
            self.fe_history = [int(self.FEvals)]
        elif self.history[-1] != float(self.fBest):
            self.history.append(float(self.fBest))
            self.fe_history.append(int(self.FEvals))

        return self.Best, float(self.fBest), self.history

