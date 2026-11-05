"""Portable entry point for the core method. No data are downloaded automatically."""
from __future__ import annotations
import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))
import numpy as np
from threadpoolctl import threadpool_limits


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_engine(protocol):
    engine = importlib.import_module(f"{protocol}_engine")
    params = json.loads((ROOT / "configs" / f"{protocol}_paper.json").read_text(encoding="utf-8"))
    engine.configure(params)
    return engine, params


def synthetic_dataset(seed=2026):
    """Artificial five-class grouped data; not ECG and not paper results."""
    rng = np.random.default_rng(seed)
    y = np.tile(np.repeat(np.arange(5), 5), 20)
    groups = np.repeat(np.array([f"synthetic_{i:02d}" for i in range(20)]), 25)
    centers = rng.normal(0, 1.5, (5, 24))
    centers[:, 8:] = 0
    X = centers[y] + rng.normal(0, 1, (y.size, 24))
    return X.astype(np.float32), y, groups, np.array([f"synthetic_feature_{i}" for i in range(24)])


def smoke_settings(params):
    # Deliberately reduced settings, never labeled as paper reproduction.
    params.update(MAXFES=24, NPOP=6, N_CHAOS=2, BLS_N1=8, BLS_N2=1, BLS_N3=12,
                  FOLDS=3, RUNS=1, PATIENT_INNER_FOLDS=3,
                  PATIENT_OUTER_SPLIT_CANDIDATES=32, PATIENT_INNER_SPLIT_CANDIDATES=32,
                  FINAL_BLS_ENSEMBLE_SIZE=2, ST_TRIGGER_PATIENCE=4)


def assert_isolation(fd, groups, patient):
    splits = [np.asarray(fd.train_idx), np.asarray(fd.val_idx), np.asarray(fd.test_idx)]
    for left, right in [(0, 1), (0, 2), (1, 2)]:
        if np.intersect1d(splits[left], splits[right]).size:
            raise ValueError("A sample occurs in two holdout partitions")
        if patient and np.intersect1d(groups[splits[left]], groups[splits[right]]).size:
            raise ValueError("A patient occurs in two holdout partitions")
    for train, val in getattr(fd, "inner_splits", ()):
        if np.intersect1d(train, val).size or np.intersect1d(groups[train], groups[val]).size:
            raise ValueError("Inner-fold patient overlap")
        if np.intersect1d(np.r_[train, val], fd.test_idx).size:
            raise ValueError("Outer test samples occur in inner CV")


def optimize(engine, protocol, params, arrays, names, ranking, inner, fold, run):
    Xtr, ytr, Xva, yva, Xte, yte = arrays
    method = "ST_FODM_MRMR_GUIDED" if protocol == "beat" else "ST_FODM_MRMR_PATIENT"
    seed_group = ("ST_FAMILY" if protocol == "beat" and params.get("ST_FAMILY_SHARED_OPTIMIZER_SEED") else method)
    seed = engine.stable_seed(params["SEED"], "optimizer", fold, run, seed_group)
    bls_seed = engine.stable_seed(params["SEED"], "bls", fold, run)
    np.random.seed(seed)
    kwargs = dict(seed=seed, bls_seed=bls_seed, mrmr_prior_enabled=True)
    if protocol == "beat":
        kwargs.update(mrmr_objective_lambda=params["ST_MRMR_GUIDED_LAMBDA"], mrmr_prior_ranking=ranking)
    else:
        kwargs.update(validation_folds=inner,
                      patient_macro_weight=params["ST_MRMR_PATIENT_MACRO_WEIGHT"],
                      mrmr_objective_lambda=engine.mrmr_objective_lambda_for_method(method))
    obj = engine.BLSFeatureSelectionObjective(Xtr, ytr, Xva, yva, names, **kwargs)
    seeds = engine.mrmr_multiscale_seed_masks(ranking, Xtr.shape[1], seed)
    optimizer_module = importlib.import_module(f"{protocol}_optimizer")
    opt = optimizer_module.FO_Direction_CEO_STAGNATION_REFINE(
        obj, Dim=Xtr.shape[1], lb=np.zeros(Xtr.shape[1]), ub=np.ones(Xtr.shape[1]),
        Np=params["NPOP"], N=params["N_CHAOS"], MaxFES=params["MAXFES"],
        alpha=params["ALPHA"], M=params["MEMORY_M"], lambda_chaos=params["LAMBDA_CHAOS"],
        min_fodm_ratio=params["ST_MIN_FODM_RATIO"],
        stagnation_trigger_patience=params["ST_TRIGGER_PATIENCE"],
        force_refine_ratio=params["ST_FORCE_REFINE_RATIO"], local_mode=params["ST_LOCAL_REFINE_MODE"],
        local_patience=params["ST_LOCAL_REFINE_PATIENCE"], initial_seed_masks=seeds)
    best_x, best_f, _ = opt.optimize()
    mask = obj.vector_to_mask(best_x)
    _, validation = obj.score_mask(mask, record=False, include_predictions=False)
    if not (0 < obj.model_fit_count <= params["MAXFES"]):
        raise RuntimeError("Effective evaluation budget invariant failed")
    before = obj.model_fit_count
    obj.score_mask(mask, record=True, include_predictions=False)
    if obj.model_fit_count != before:
        raise RuntimeError("A cached mask consumed an effective evaluation")
    # Keep the cache check out of the reported optimization history.
    obj.call_count -= 1
    obj.eval_history.pop()
    return mask, validation, obj, opt, seed, bls_seed


def run_experiment(args):
    engine, params = load_engine(args.protocol)
    is_demo = args.command == "demo"
    if is_demo:
        smoke_settings(params)
        engine.configure(params)
        X, y, records, names = synthetic_dataset()
        data_info = {"source": "synthetic", "notice": "Artificial data; not ECG or paper results"}
    else:
        feature_file = args.features.resolve()
        # Original NPZ format may contain object strings. Use only trusted locally created NPZs.
        with np.load(feature_file, allow_pickle=True) as z:
            if "groups" not in z.files:
                raise ValueError("NPZ must contain genuine record IDs in 'groups'")
            if z["X"].shape[1] != 262:
                raise ValueError("Expected 262 base columns (256 waveform + 6 RR); HFAE is appended internally")
        X, y, records, names, data_info = engine.load_feature_dataset(feature_file)
        if args.protocol == "patient":
            X, y, records, names, data_info = engine.apply_protocol_dataset_filter(X, y, records, names, data_info)
        if X.shape[1] != 269 or not np.all(np.isfinite(X)):
            raise ValueError("Expected a finite 269-column ECG/RR/HFAE feature matrix")
    groups = engine.make_patient_groups(records) if args.protocol == "patient" else records
    split_kwargs = {"record_groups": records} if args.protocol == "patient" else {}
    folds = engine.make_folds(X, y, groups, int(params["FOLDS"]), int(params["SEED"]), **split_kwargs)
    for fd in folds:
        assert_isolation(fd, groups, args.protocol == "patient")
    chosen = folds if args.all else [fd for fd in folds if fd.fold == args.fold]
    if not chosen:
        raise ValueError("Requested fold does not exist")
    if args.command == "run" and not 1 <= args.run <= params["RUNS"]:
        raise ValueError("Run index outside paper configuration")
    # New output folders prevent accidental replacement of previous experiments.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "effective_config.json", params)
    import scipy, sklearn, pandas
    write_json(output / "environment.json", dict(python=platform.python_version(), numpy=np.__version__,
               scipy=scipy.__version__, sklearn=sklearn.__version__, pandas=pandas.__version__))
    write_json(output / "dataset.json", {"profile": "synthetic_demo" if is_demo else "paper_core",
               "protocol": args.protocol, "shape": X.shape, "class_counts": dict(zip(*np.unique(y, return_counts=True))),
               "classes": ["synthetic_0", "synthetic_1", "synthetic_2", "synthetic_3", "synthetic_4"] if is_demo
                   else (["N", "S", "V", "F", "Q"] if args.protocol == "beat" else ["N", "A", "V", "L", "R"]),
               "actual_outer_folds": len(folds), "feature_names": names, "data_info": data_info,
               "input_sha256": None if is_demo else engine.file_sha256(args.features)})
    rows = []
    for fd in chosen:
        arrays = engine.prepare_fold_arrays(X, y, fd)
        inner = engine.prepare_inner_cv_arrays(X, y, fd, patient_groups=groups) if args.protocol == "patient" else []
        ranking = engine.mrmr_ranking(arrays[0], arrays[1], engine.stable_seed(params["SEED"], fd.fold, "MRMR", "ranking"))
        split = {"train": fd.train_idx, "validation": fd.val_idx, "test": fd.test_idx, "ranking": ranking}
        for i, (tr, va) in enumerate(getattr(fd, "inner_splits", ()), 1):
            split[f"inner_{i}_train"], split[f"inner_{i}_validation"] = tr, va
        np.savez_compressed(output / f"fold_{fd.fold:02d}_splits.npz", **split)
        runs = range(1, params["RUNS"] + 1) if args.all else [args.run]
        for run in runs:
            mask, val, obj, opt, seed, bls_seed = optimize(engine, args.protocol, params, arrays, names, ranking, inner, fd.fold, run)
            Xtr, ytr, Xva, yva, Xte, yte = arrays
            kwargs = {"test_patient_groups": groups[fd.test_idx]} if args.protocol == "patient" else {}
            test = engine.evaluate_selected_mask_on_holdout_test(Xtr, ytr, Xva, yva, Xte, yte, names, mask,
                                                                seed, bls_seed=bls_seed, **kwargs)
            tag = f"F{fd.fold:02d}_R{run:02d}"
            np.savez_compressed(output / f"{tag}_predictions.npz", mask=mask, y_true=test["TrueLabels"],
                                y_pred=test["PredLabels"], scores=test["ScoreMatrix"])
            write_json(output / f"{tag}_history.json", obj.eval_history)
            row = dict(fold=fd.fold, run=run, optimizer_seed=seed, bls_seed=bls_seed,
                       selected_count=int(mask.sum()), selected_indices=np.flatnonzero(mask),
                       selected_feature_names=names[mask], effective_fes=obj.model_fit_count,
                       actual_search_bls_fits=getattr(obj, "actual_bls_fit_count", obj.model_fit_count),
                       raw_objective_calls=obj.call_count, budget_requested=params["MAXFES"],
                       unused_budget=params["MAXFES"]-obj.model_fit_count,
                       switch_reason=opt.switch_trigger_reason, switch_fes=opt.switch_fevals,
                       local_fes=opt.local_refine_fevals, budget_fill_fes=opt.budget_fill_model_fits,
                       validation_fitness=val["Fitness"], validation_macro_f1=val["MacroF1"],
                       test_macro_f1=test["MacroF1"], test_accuracy=test["Accuracy"],
                       final_head_fits=test["FinalHeadFits"],
                       prior_alpha=test["FinalPriorCorrectionAlpha"],
                       test_class_metrics=test["ClassMetrics"])
            write_json(output / f"{tag}_result.json", row)
            rows.append(row)
            print(f"{tag}: FES={obj.model_fit_count}/{params['MAXFES']}, selected={mask.sum()}, test Macro-F1={test['MacroF1']:.6f}", flush=True)
    write_json(output / "summary.json", {"profile": "synthetic_demo" if is_demo else "paper_core", "runs": rows})
    print(f"Saved {len(rows)} run(s) to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ["demo", "run"]:
        p = subs.add_parser(command)
        p.add_argument("--protocol", choices=["beat", "patient"], default="beat")
        p.add_argument("--output", type=Path, required=True, help="New folder; existing folders are rejected")
        p.add_argument("--fold", type=int, default=1)
        p.add_argument("--run", type=int, default=1)
        p.add_argument("--all", action="store_true", help="All configured folds and runs; substantial compute on real data")
        p.add_argument("--threads", type=int, default=2)
        if command == "run":
            p.add_argument("--features", type=Path, required=True)
    p = subs.add_parser("prepare", help="Build the original base NPZ from a local MIT-BIH WFDB directory")
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", choices=["beat", "patient"], default="beat")
    args = parser.parse_args()
    if args.command == "prepare":
        from mitbih_raw_loader import build_mitbih_feature_npz
        _, params = load_engine(args.protocol)
        if args.output.exists():
            raise FileExistsError("Choose a new NPZ output file")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        build_mitbih_feature_npz(args.raw_dir.resolve(), args.output.resolve(), SimpleNamespace(**params), force=True)
    else:
        if args.threads < 1:
            parser.error("--threads must be positive")
        with threadpool_limits(limits=args.threads):
            run_experiment(args)


if __name__ == "__main__":
    main()
