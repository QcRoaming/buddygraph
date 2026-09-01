# RUN: %PYTHON %s %buddygraph_src_root %buddygraph_obj_root %llvm_tools_dir %t | FileCheck %s
# CHECK: reference=5.000000e-01
# CHECK: unoptimized=5.000000e-01 max_abs_error=0.000000e+00 max_rel_error=0.000000e+00
# CHECK: optimized=5.000000e-01 max_abs_error=0.000000e+00 max_rel_error=0.000000e+00
# CHECK: nan_unoptimized=nan nan_optimized=nan
# CHECK: fusion=verified finite=true nan=true
# CHECK: status=PASS

"""Compile ONNX Clip fixtures through direct and fused Clamp paths."""

from __future__ import annotations

import math
import pathlib
import re
import subprocess
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def compile_and_run(
    optimizer: pathlib.Path,
    runner: pathlib.Path,
    source: pathlib.Path,
    output: pathlib.Path,
    optimized: bool,
) -> float:
    pipeline = []
    if optimized:
        pipeline.extend(["--canonicalize", "--cse", "--bgraph-fuse-elementwise"])
    pipeline.extend(
        [
            "--convert-bgraph-to-linalg",
            "--one-shot-bufferize=bufferize-function-boundaries",
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--convert-scf-to-cf",
            "--convert-cf-to-llvm",
            "--convert-math-to-llvm",
            "--convert-arith-to-llvm",
            "--finalize-memref-to-llvm",
            "--convert-func-to-llvm",
            "--reconcile-unrealized-casts",
        ]
    )
    run([str(optimizer), *pipeline, str(source), "-o", str(output)])
    stdout = run([str(runner), str(output), "-e", "main", "-entry-point-result=f32"])
    if re.search(r"\bnan\b", stdout, re.IGNORECASE):
        return math.nan
    match = re.search(r"[-+]?\d+(?:\.\d+)?e[-+]?\d+", stdout, re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"mlir-runner produced no floating-point result: {stdout!r}")
    return float(match.group(0))


def import_model(
    source_root: pathlib.Path,
    model: pathlib.Path,
    output: pathlib.Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(source_root / "frontend/BGraph/import_onnx.py"),
            str(model),
            "--function-name",
            "main",
            "--scalar-return",
            "-o",
            str(output),
        ],
        check=True,
    )


def require_fused_clamp(optimizer: pathlib.Path, source: pathlib.Path) -> None:
    fused = run([str(optimizer), "--bgraph-fuse-elementwise", str(source)])
    if '"bgraph.fused_elementwise"' not in fused or '"bgraph.clamp"' in fused:
        raise RuntimeError("the optimized Clamp fixture did not form a fused elementwise op")


def main() -> None:
    source_root = pathlib.Path(sys.argv[1])
    object_root = pathlib.Path(sys.argv[2])
    llvm_tools = pathlib.Path(sys.argv[3])
    temp_root = pathlib.Path(sys.argv[4])
    models = temp_root.with_suffix(".models")
    subprocess.run(
        [sys.executable, str(source_root / "frontend/BGraph/generate_test_models.py"), str(models)],
        check=True,
    )

    model_path = models / "clamp_constant.onnx"
    imported = temp_root.with_suffix(".mlir")
    import_model(source_root, model_path, imported)
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in onnx.load(model_path).graph.initializer
    }
    reference = float(
        np.mean(
            np.clip(
                initializers["a"] + initializers["b"],
                initializers["clip_min"],
                initializers["clip_max"],
            )
            * initializers["factor"]
        )
    )

    optimizer = object_root / "bin/buddygraph-opt"
    runner = llvm_tools / "mlir-runner"
    require_fused_clamp(optimizer, imported)
    results = {
        "unoptimized": compile_and_run(
            optimizer, runner, imported, temp_root.with_suffix(".unoptimized.llvm.mlir"), False
        ),
        "optimized": compile_and_run(
            optimizer, runner, imported, temp_root.with_suffix(".optimized.llvm.mlir"), True
        ),
    }

    print(f"reference={reference:.6e}")
    tolerance = 1.0e-6
    for name, result in results.items():
        absolute_error = abs(result - reference)
        relative_error = absolute_error / max(abs(reference), 1.0e-12)
        print(
            f"{name}={result:.6e} max_abs_error={absolute_error:.6e} "
            f"max_rel_error={relative_error:.6e}"
        )
        if absolute_error > tolerance and relative_error > tolerance:
            raise RuntimeError(f"{name} result is outside tolerance")

    nan_imported = temp_root.with_suffix(".nan.mlir")
    import_model(source_root, models / "clamp_nan.onnx", nan_imported)
    require_fused_clamp(optimizer, nan_imported)
    nan_results = {
        name: compile_and_run(
            optimizer,
            runner,
            nan_imported,
            temp_root.with_suffix(f".nan.{name}.llvm.mlir"),
            optimized,
        )
        for name, optimized in (("unoptimized", False), ("optimized", True))
    }
    if not all(math.isnan(value) for value in nan_results.values()):
        raise RuntimeError(f"Clamp did not preserve NaN: {nan_results}")
    print("nan_unoptimized=nan nan_optimized=nan")
    print("fusion=verified finite=true nan=true")
    print("status=PASS")


if __name__ == "__main__":
    main()
