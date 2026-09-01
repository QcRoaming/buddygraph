#!/usr/bin/env python3
"""Reproduce BuddyGraph compile-time, IR-structure, and runner measurements."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Sequence


PROFILES: dict[str, list[str]] = {
    "off": [],
    "canonicalize_cse": ["--canonicalize", "--cse"],
    "bn_folding": ["--bgraph-fold-bn-into-conv"],
    "elementwise_fusion": ["--bgraph-fuse-elementwise"],
    "all": [
        "--bgraph-fold-bn-into-conv",
        "--canonicalize",
        "--cse",
        "--bgraph-fuse-elementwise",
    ],
}

LOWERING = [
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


def run(command: Sequence[str], environment: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    return completed.stdout


def sample(command: Sequence[str], repetitions: int, environment: dict[str, str] | None = None) -> list[float]:
    samples: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        run(command, environment)
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def summarize(samples: Sequence[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
    }


def static_alloc_bytes(text: str) -> int:
    total = 0
    for line in text.splitlines():
        if "memref.alloc" not in line:
            continue
        match = re.search(
            r"memref<(?:(?P<shape>[0-9]+(?:x[0-9]+)*)x)?f32>", line
        )
        if match:
            shape = match.group("shape")
            elements = (
                math.prod(int(dimension) for dimension in shape.split("x"))
                if shape
                else 1
            )
            total += elements * 4
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path(__file__).parents[1])
    parser.add_argument("--build-root", type=pathlib.Path, default=None)
    parser.add_argument("--llvm-tools", type=pathlib.Path, default=pathlib.Path("/buddy-mlir/llvm/build/bin"))
    parser.add_argument("--compile-repetitions", type=int, default=7)
    parser.add_argument("--runtime-warmup", type=int, default=3)
    parser.add_argument("--runtime-repetitions", type=int, default=20)
    parser.add_argument("-o", "--output", type=pathlib.Path)
    arguments = parser.parse_args()
    source_root = arguments.source_root.resolve()
    build_root = (arguments.build_root or source_root / "build").resolve()
    optimizer = build_root / "bin/buddygraph-opt"
    runner = arguments.llvm_tools.resolve() / "mlir-runner"
    python_path = os.pathsep.join(
        [
            str(source_root / ".deps"),
            str(arguments.llvm_tools.resolve().parent / "tools/mlir/python_packages/mlir_core"),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    environment = dict(os.environ, PYTHONPATH=python_path)

    with tempfile.TemporaryDirectory(prefix="buddygraph-benchmark-") as temporary:
        temp = pathlib.Path(temporary)
        models = temp / "models"
        generator = source_root / "frontend/BGraph/generate_test_models.py"
        importer = source_root / "frontend/BGraph/import_onnx.py"
        run([sys.executable, str(generator), str(models)], environment)
        model = models / "elementwise_constant.onnx"
        imported = temp / "model.mlir"
        import_command = [
            sys.executable,
            str(importer),
            str(model),
            "--function-name",
            "main",
            "--scalar-return",
            "-o",
            str(imported),
        ]
        import_timing = summarize(
            sample(import_command, arguments.compile_repetitions, environment)
        )
        run(import_command, environment)

        pass_commands = {
            "shape_inference": [str(optimizer), "--bgraph-infer-shapes", str(imported), "-o", os.devnull],
            "bn_folding": [str(optimizer), "--bgraph-fold-bn-into-conv", str(imported), "-o", os.devnull],
            "elementwise_fusion": [str(optimizer), "--bgraph-fuse-elementwise", str(imported), "-o", os.devnull],
        }
        pass_timing = {
            name: summarize(sample(command, arguments.compile_repetitions))
            for name, command in pass_commands.items()
        }

        ir_outputs: dict[str, str] = {}
        for name, profile in {"off": PROFILES["off"], "all": PROFILES["all"]}.items():
            bgraph_output = temp / f"{name}.bgraph.mlir"
            linalg_output = temp / f"{name}.linalg.mlir"
            buffer_output = temp / f"{name}.buffer.mlir"
            run([str(optimizer), *profile, str(imported), "-o", str(bgraph_output)])
            run([str(optimizer), *profile, "--convert-bgraph-to-linalg", str(imported), "-o", str(linalg_output)])
            run(
                [
                    str(optimizer),
                    *profile,
                    "--convert-bgraph-to-linalg",
                    "--one-shot-bufferize=bufferize-function-boundaries",
                    str(imported),
                    "-o",
                    str(buffer_output),
                ]
            )
            ir_outputs[f"{name}_bgraph"] = bgraph_output.read_text(encoding="utf-8")
            ir_outputs[f"{name}_linalg"] = linalg_output.read_text(encoding="utf-8")
            ir_outputs[f"{name}_buffer"] = buffer_output.read_text(encoding="utf-8")

        ir_metrics = {}
        for name in ("off", "all"):
            bgraph = ir_outputs[f"{name}_bgraph"]
            linalg = ir_outputs[f"{name}_linalg"]
            buffer = ir_outputs[f"{name}_buffer"]
            ir_metrics[name] = {
                "bgraph_ops": len(re.findall(r'"bgraph\.[a-z0-9_]+"', bgraph)),
                "fused_elementwise_ops": bgraph.count('"bgraph.fused_elementwise"'),
                "linalg_generic_ops": linalg.count("linalg.generic"),
                "linalg_named_conv_ops": linalg.count("linalg.conv_2d_nchw_fchw"),
                "memref_alloc_ops": buffer.count("memref.alloc"),
                "static_alloc_bytes": static_alloc_bytes(buffer),
            }

        runtime: dict[str, dict[str, float]] = {}
        pipeline_compile: dict[str, dict[str, float]] = {}
        for name, profile in PROFILES.items():
            executable = temp / f"{name}.llvm.mlir"
            compile_command = [str(optimizer), *profile, *LOWERING, str(imported), "-o", str(executable)]
            pipeline_compile[name] = summarize(
                sample(compile_command, arguments.compile_repetitions)
            )
            run(compile_command)
            runner_command = [str(runner), str(executable), "-e", "main", "-entry-point-result=f32"]
            for _ in range(arguments.runtime_warmup):
                run(runner_command)
            values = sample(runner_command, arguments.runtime_repetitions)
            result_text = run(runner_command).strip()
            if not result_text.startswith("7.000000e+00"):
                raise RuntimeError(f"profile {name} returned unexpected result {result_text!r}")
            runtime[name] = summarize(values)

        report = {
            "environment": {
                "platform": platform.platform(),
                "processor": platform.processor() or "not reported by Python",
                "python": platform.python_version(),
                "optimizer": str(optimizer),
                "model": "elementwise_constant.onnx",
                "input_shape": [1, 1, 2, 2],
                "compile_repetitions": arguments.compile_repetitions,
                "runtime_warmup": arguments.runtime_warmup,
                "runtime_repetitions": arguments.runtime_repetitions,
                "runtime_scope": "separate mlir-runner process including parse and JIT startup",
            },
            "compile_time": {
                "importer": import_timing,
                "passes": pass_timing,
                "full_pipeline_by_profile": pipeline_compile,
            },
            "ir": ir_metrics,
            "runtime": runtime,
        }
        serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if arguments.output:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(serialized, encoding="utf-8")
        sys.stdout.write(serialized)


if __name__ == "__main__":
    main()
