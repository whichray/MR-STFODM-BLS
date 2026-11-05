"""Check source integrity, portability, reproducibility and effective-budget accounting."""
from __future__ import annotations
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def body_text(source, node):
    start = min([node.lineno] + [n.lineno for n in getattr(node, "decorator_list", [])])
    return "".join(source.splitlines(keepends=True)[start - 1:node.end_lineno])

def main():
    provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    for filename, expected in provenance["unchanged_dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / "core" / filename).read_bytes()).hexdigest() == expected, filename
    for protocol in ["beat", "patient"]:
        for suffix, field in [("engine", "engine_bodies"), ("optimizer", "optimizer_bodies")]:
            source = (ROOT / "core" / f"{protocol}_{suffix}.py").read_text(encoding="utf-8")
            nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            for name, expected in provenance["sources"][protocol][field].items():
                assert hashlib.sha256(body_text(source, nodes[name]).encode()).hexdigest() == expected, name
    # A separate cwd and fresh output folders also check path independence.
    with tempfile.TemporaryDirectory(prefix="mr_stfodm_verify_") as temporary:
        tmp = Path(temporary)
        for protocol in ["beat", "patient"]:
            summaries = []
            predictions = []
            for repeat in [1, 2]:
                output = tmp / f"{protocol}_{repeat}"
                subprocess.run([sys.executable, "-B", str(ROOT / "run.py"), "demo", "--protocol", protocol,
                                "--output", str(output)], cwd=tmp, check=True, capture_output=True, text=True)
                summaries.append(json.loads((output / "summary.json").read_text(encoding="utf-8")))
                with np.load(output / "F01_R01_predictions.npz", allow_pickle=False) as z:
                    predictions.append({k: z[k].copy() for k in z.files})
                row = summaries[-1]["runs"][0]
                assert row["effective_fes"] == row["budget_requested"] == 24
                assert row["unused_budget"] == 0
                expected_fits = 24 if protocol == "beat" else 72
                assert row["actual_search_bls_fits"] == expected_fits
                history = json.loads((output / "F01_R01_history.json").read_text(encoding="utf-8"))
                fes = [r["EffectiveFEval"] for r in history]
                assert all(a <= b for a, b in zip(fes, fes[1:])) and max(fes) == 24
                with np.load(output / "fold_01_splits.npz", allow_pickle=False) as splits:
                    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
                        assert not np.intersect1d(splits[left], splits[right]).size
            assert summaries[0] == summaries[1], f"Nondeterministic summary: {protocol}"
            for key in predictions[0]:
                np.testing.assert_array_equal(predictions[0][key], predictions[1][key])
            print(f"PASS {protocol}: deterministic outputs, source-body integrity, split isolation, FES and actual fits")
    print("PASS: core-package checks. These checks do not reproduce the manuscript's full experiments.")

if __name__ == "__main__":
    main()
