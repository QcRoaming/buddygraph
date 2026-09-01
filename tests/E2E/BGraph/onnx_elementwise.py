# RUN: %PYTHON %s %buddygraph_src_root %buddygraph_obj_root %llvm_tools_dir %t | FileCheck %s
# CHECK: reference=7.000000e+00
# CHECK: unoptimized=7.000000e+00 max_abs_error=0.000000e+00 max_rel_error=0.000000e+00
# CHECK: optimized=7.000000e+00 max_abs_error=0.000000e+00 max_rel_error=0.000000e+00
# CHECK: status=PASS

"""Compile an ONNX fixture through both pipelines and compare with NumPy."""

from __future__ import annotations

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
        pipeline.extend(
            [
                "--bgraph-fold-bn-into-conv",
                "--canonicalize",
                "--cse",
                "--bgraph-fuse-elementwise",
            ]
        )
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
    match = re.search(r"[-+]?\d+(?:\.\d+)?e[-+]?\d+", stdout, re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"mlir-runner produced no floating-point result: {stdout!r}")
    return float(match.group(0))


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
    model_path = models / "elementwise_constant.onnx"
    imported = temp_root.with_suffix(".mlir")
    subprocess.run(
        [
            sys.executable,
            str(source_root / "frontend/BGraph/import_onnx.py"),
            str(model_path),
            "--function-name",
            "main",
            "--scalar-return",
            "-o",
            str(imported),
        ],
        check=True,
    )

    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in onnx.load(model_path).graph.initializer
    }
    convolution = initializers["a"] * initializers["w"].reshape(1, 1, 1, 1)
    convolution = convolution + initializers["conv_bias"].reshape(1, 1, 1, 1)
    normalized = (
        (convolution - initializers["bn_mean"].reshape(1, 1, 1, 1))
        * initializers["bn_scale"].reshape(1, 1, 1, 1)
        / np.sqrt(initializers["bn_variance"].reshape(1, 1, 1, 1) + 1.0e-5)
        + initializers["bn_bias"].reshape(1, 1, 1, 1)
    )
    reference = float(
        np.mean(np.maximum(normalized + initializers["b"], 0.0) * initializers["factor"])
    )
    optimizer = object_root / "bin/buddygraph-opt"
    runner = llvm_tools / "mlir-runner"
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
    print("status=PASS")


if __name__ == "__main__":
    main()
