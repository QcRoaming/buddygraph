#!/usr/bin/env python3
"""Measure single-thread native BuddyGraph execution, excluding compilation.

Source scripts/env.sh before running. Requires the LLVM SDK's mlir-translate,
opt and llc, a C compiler, and NumPy. Each kernel accepts runtime inputs and a
caller-owned output. Intermediate malloc/free operations remain inside timing.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np

from benchmark import LOWERING, PROFILES, run, static_alloc_bytes, summarize

NATIVE_PROFILES = {
    **PROFILES,
    "direct_output": PROFILES["all"],
    "cpu": PROFILES["all"],
}


BATCH_SOURCE = r"""
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <time.h>
typedef void (*kernel_fn)(void *, void *, void *, void *);
double run_batch(kernel_fn kernel, void *x, void *b, void *factor, void *out,
                 int64_t iterations) {
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int64_t i = 0; i < iterations; ++i)
    kernel(x, b, factor, out);
  clock_gettime(CLOCK_MONOTONIC, &end);
  return (end.tv_sec - start.tv_sec) * 1000.0
       + (end.tv_nsec - start.tv_nsec) / 1000000.0;
}
"""


def tensor(shape: tuple[int, ...]) -> str:
    return "tensor<" + "x".join(map(str, shape)) + "xf32>"


def memref(shape: tuple[int, ...]) -> str:
    return tensor(shape).replace("tensor", "memref")


def dense(value: np.ndarray) -> str:
    if value.ndim == 0:
        return f"{float(value):.9e}"
    return "[" + ", ".join(dense(item) for item in value) + "]"


def workload(kind: str, size: int, *, in_channels: int = 8,
             out_channels: int = 16, kernel: int = 3, stride: int = 1,
             dilation: int = 1) -> dict:
    rng = np.random.default_rng(20261105 + size)
    padding = dilation * (kernel - 1) // 2
    output_size = (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    xshape = (1, in_channels, size, size) if kind == "conv_bn" else (size,)
    yshape = (1, out_channels, output_size, output_size) if kind == "conv_bn" else (size,)
    xt, yt = tensor(xshape), tensor(yshape)
    lines = [f"func.func private @compute(%x: {xt}, %b: {yt}, %factor: {yt}) -> {yt} {{"]
    constants = {}
    source = "%x"
    if kind == "conv_bn":
        constants = {
            "w": rng.uniform(-0.1, 0.1, (out_channels, in_channels, kernel, kernel)).astype(np.float32),
            "bias": rng.uniform(-0.1, 0.1, out_channels).astype(np.float32),
            "scale": rng.uniform(0.5, 1.5, out_channels).astype(np.float32),
            "beta": rng.uniform(-0.2, 0.2, out_channels).astype(np.float32),
            "mean": rng.uniform(-0.1, 0.1, out_channels).astype(np.float32),
            "var": rng.uniform(0.5, 1.5, out_channels).astype(np.float32),
        }
        for name, value in constants.items():
            lines.append(f"  %{name} = arith.constant dense<{dense(value)}> : {tensor(value.shape)}")
        wt, ct = tensor(constants["w"].shape), tensor((out_channels,))
        lines += [
            '  %conv = "bgraph.conv2d"(%x, %w, %bias) {'
            f'dilations = array<i64: {dilation}, {dilation}>, '
            'groups = 1 : i64, layout = #bgraph.layout<nchw>, '
            f'pads = array<i64: {padding}, {padding}, {padding}, {padding}>, '
            f'strides = array<i64: {stride}, {stride}>}}'
            f" : ({xt}, {wt}, {ct}) -> {yt}",
            '  %bn = "bgraph.batch_norm"(%conv, %scale, %beta, %mean, %var) '
            '{epsilon = 1.0e-5 : f64, layout = #bgraph.layout<nchw>}'
            f" : ({yt}, {ct}, {ct}, {ct}, {ct}) -> {yt}",
        ]
        source = "%bn"
    lines.append(f'  %sum = "bgraph.add"({source}, %b) : ({yt}, {yt}) -> {yt}')
    if kind == "clamp":
        lines.append('  %act = "bgraph.clamp"(%sum) '
                     '{min_value = -0.5 : f32, max_value = 0.75 : f32}'
                     f" : ({yt}) -> {yt}")
    else:
        lines.append(f'  %act = "bgraph.relu"(%sum) : ({yt}) -> {yt}')
    lines += [
        f'  %result = "bgraph.mul"(%act, %factor) : ({yt}, {yt}) -> {yt}',
        f"  return %result : {yt}", "}",
    ]
    wrapper = "\n".join([
        f"func.func @kernel(%xm: {memref(xshape)}, %bm: {memref(yshape)}, "
        f"%fm: {memref(yshape)}, %out: {memref(yshape)}) "
        "attributes {llvm.emit_c_interface} {",
        f"  %x = bufferization.to_tensor %xm restrict : {memref(xshape)} to {xt}",
        f"  %b = bufferization.to_tensor %bm restrict : {memref(yshape)} to {yt}",
        f"  %factor = bufferization.to_tensor %fm restrict : {memref(yshape)} to {yt}",
        f"  %result = func.call @compute(%x, %b, %factor) : ({xt}, {yt}, {yt}) -> {yt}",
        '  bufferization.materialize_in_destination %result in restrict writable %out'
        f" : ({yt}, {memref(yshape)}) -> ()",
        "  return", "}",
    ]) + "\n"
    return {"kind": kind, "size": size, "xshape": xshape, "yshape": yshape,
            "constants": constants, "ir": "\n".join(lines) + "\n", "wrapper": wrapper,
            "padding": padding, "stride": stride, "dilation": dilation}


def inputs_and_reference(case: dict, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, case["xshape"]).astype(np.float32)
    b = rng.uniform(-0.5, 0.5, case["yshape"]).astype(np.float32)
    factor = rng.uniform(0.25, 1.75, case["yshape"]).astype(np.float32)
    value = x
    if case["kind"] == "conv_bn":
        c = case["constants"]
        pad, stride, dilation = case["padding"], case["stride"], case["dilation"]
        padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        effective_kernel = dilation * (c["w"].shape[-1] - 1) + 1
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, (effective_kernel, effective_kernel), axis=(2, 3))
        windows = windows[:, :, ::stride, ::stride, ::dilation, ::dilation]
        value = np.einsum("nchwij,ocij->nohw", windows, c["w"], optimize=False)
        channel = lambda name: c[name].reshape(1, -1, 1, 1)
        value = value + channel("bias")
        value = (value - channel("mean")) * channel("scale") / np.sqrt(
            channel("var") + np.float32(1.0e-5)) + channel("beta")
    value = value + b
    value = np.clip(value, -0.5, 0.75) if case["kind"] == "clamp" else np.maximum(value, 0)
    return [x, b, factor, np.empty(case["yshape"], dtype=np.float32)], value * factor


def descriptor(array: np.ndarray) -> ctypes.Structure:
    rank = array.ndim

    class MemRef(ctypes.Structure):
        _fields_ = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p),
                    ("offset", ctypes.c_int64), ("sizes", ctypes.c_int64 * rank),
                    ("strides", ctypes.c_int64 * rank)]

    return MemRef(array.ctypes.data, array.ctypes.data, 0,
                  (ctypes.c_int64 * rank)(*array.shape),
                  (ctypes.c_int64 * rank)(*(s // array.itemsize for s in array.strides)))


def compile_kernel(case: dict, profile: str, directory: Path, optimizer: Path,
                   llvm: Path, cc: str, vector_width: int = 8,
                   channel_tile: int = 4) -> tuple[ctypes.CDLL, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "input.mlir"
    source.write_text(case["ir"], encoding="utf-8")
    buffer = directory / "buffer.mlir"
    linalg = directory / "linalg.mlir"
    adapted = directory / "adapted.mlir"
    scheduled = directory / "scheduled.mlir"
    lowered = directory / "llvm.mlir"
    llvm_ir = directory / "kernel.ll"
    optimized = directory / "kernel.opt.ll"
    obj = directory / "kernel.o"
    library = directory / "kernel.so"
    start = time.perf_counter()
    run([str(optimizer), *NATIVE_PROFILES[profile], LOWERING[0], str(source), "-o", str(linalg)])
    # The BGraph full-conversion target intentionally rejects Bufferization ops.
    # Add the caller-owned output adapter after that conversion, then inline it.
    text = linalg.read_text(encoding="utf-8")
    closing = text.rfind("}")
    adapted.write_text(text[:closing] + case["wrapper"] + text[closing:], encoding="utf-8")
    buffer_pipeline = (["--bgraph-bufferize"] if profile in ("direct_output", "cpu")
                       else ["--inline", "--symbol-dce", LOWERING[1],
                             "--buffer-deallocation-pipeline"])
    run([str(optimizer), *buffer_pipeline, str(adapted), "-o", str(buffer)])
    schedule_pipeline = ([f"--bgraph-vectorize-conv=vector-width={vector_width} channel-tile={channel_tile}",
                          "--canonicalize"] if profile == "cpu" else [])
    run([str(optimizer), *schedule_pipeline, str(buffer), "-o", str(scheduled)])
    lowering = [LOWERING[2], "--expand-strided-metadata", *LOWERING[3:7],
                "--convert-vector-to-llvm", *LOWERING[7:]]
    run([str(optimizer), *lowering, str(scheduled), "-o", str(lowered)])
    run([str(llvm / "mlir-translate"), "--mlir-to-llvmir", str(lowered), "-o", str(llvm_ir)])
    run([str(llvm / "opt"), "-passes=default<O3>", "-S", str(llvm_ir), "-o", str(optimized)])
    run([str(llvm / "llc"), "-O3", "-relocation-model=pic", "-filetype=obj",
         str(optimized), "-o", str(obj)])
    runtime_lib = llvm.parent / "lib"
    run([cc, "-shared", "-Wl,-z,defs", str(obj),
         str(runtime_lib / "libmlir_c_runner_utils.so"),
         f"-Wl,-rpath,{runtime_lib}", "-lm", "-o", str(library)])
    compile_ms = (time.perf_counter() - start) * 1000
    text = buffer.read_text(encoding="utf-8")
    allocs = text.count("memref.alloc(")
    deallocs = text.count("memref.dealloc ")
    if allocs != deallocs:
        raise RuntimeError(f"Unbalanced allocation lifetime in {buffer}: {allocs} / {deallocs}")
    scheduled_text = scheduled.read_text(encoding="utf-8")
    metrics = {"native_build_wall_ms": compile_ms, "memref_alloc_ops": allocs,
               "memref_dealloc_ops": deallocs, "static_alloc_bytes": static_alloc_bytes(text),
               "linalg_generic_ops": text.count("linalg.generic"),
               "linalg_conv_ops": text.count("linalg.conv_2d_nchw_fchw"),
               "memref_copy_ops": text.count("memref.copy"),
               "scheduled_vector_load_ops": scheduled_text.count("vector.load"),
               "scheduled_linalg_conv_ops": scheduled_text.count("linalg.conv_2d_nchw_fchw"),
               "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    return ctypes.CDLL(str(library)), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm-tools", type=Path, default=Path(os.environ.get("LLVM_BUILD", "")) / "bin")
    parser.add_argument("--optimizer", type=Path, default=Path(__file__).resolve().parents[1] / "build/bin/buddygraph-opt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=31)
    parser.add_argument("--batch-ms", type=float, default=20)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 65536, 1048576])
    parser.add_argument("--conv-sizes", type=int, nargs="*", default=[16, 32])
    parser.add_argument("--profiles", nargs="+", choices=NATIVE_PROFILES, default=list(PROFILES))
    parser.add_argument("--vector-width", type=int, default=8)
    parser.add_argument("--channel-tile", type=int, default=4)
    args = parser.parse_args()
    if args.rounds < 1 or args.batch_ms <= 0 or any(n <= 0 for n in args.sizes + args.conv_sizes):
        parser.error("rounds, batch-ms and sizes must be positive")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    llvm = args.llvm_tools.resolve()
    optimizer = args.optimizer.resolve()
    shim_source = output / "batch.c"
    shim_source.write_text(BATCH_SOURCE, encoding="utf-8")
    shim_path = output / "batch.so"
    run([args.cc, "-O3", "-shared", "-fPIC", str(shim_source), "-o", str(shim_path)])
    shim = ctypes.CDLL(str(shim_path))
    shim.run_batch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64]
    shim.run_batch.restype = ctypes.c_double
    cpuinfo = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                      if line.startswith("model name")), platform.processor())
    report = {"environment": {
        "platform": platform.platform(), "cpu": cpu_model, "python": platform.python_version(),
        "numpy": np.__version__, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "llvm_version": run([str(llvm / "llc"), "--version"]).splitlines()[0],
        "cc_version": run([args.cc, "--version"]).splitlines()[0],
        "optimizer_sha256": hashlib.sha256(optimizer.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "threads": 1, "rounds": args.rounds, "target_batch_ms": args.batch_ms,
        "vector_width": args.vector_width, "channel_tile": args.channel_tile,
        "warmup_batches": 5, "correctness_seeds": [11, 22, 33],
        "rtol": 2e-4, "atol": 2e-5,
        "scope": "Native AOT function call including intermediate malloc/free; excludes input/output allocation, compilation, Python and process startup. LLVM opt O3, llc O3 generic x86-64, no fast-math.",
        "sampling": "C loop, adaptive iterations; per-call batch means, randomized profile order each round; p95 is p95 of batch means, not individual call tail latency.",
    }, "workloads": {}}
    cases = [workload("relu", n) for n in args.sizes]
    cases += [workload("clamp", args.sizes[-1])]
    cases += [workload("conv_bn", n) for n in args.conv_sizes]
    for case in cases:
        name = f"{case['kind']}_{case['size']}"
        print(f"Building {name}", flush=True)
        records, handles, kernels, iters, samples = {}, {}, {}, {}, {}
        arrays, reference = inputs_and_reference(case, 33)
        descriptors = [descriptor(array) for array in arrays]
        pointers = [ctypes.byref(item) for item in descriptors]
        for profile in args.profiles:
            handles[profile], records[profile] = compile_kernel(
                case, profile, output / name / profile, optimizer, llvm, args.cc,
                args.vector_width, args.channel_tile)
            function = handles[profile]._mlir_ciface_kernel
            function.argtypes = [ctypes.c_void_p] * 4
            function.restype = None
            kernels[profile] = function
            abs_errors = []
            for seed in [11, 22, 33]:
                check_arrays, expected = inputs_and_reference(case, seed)
                check_descriptors = [descriptor(array) for array in check_arrays]
                check_arrays[-1].fill(np.nan)
                function(*(ctypes.byref(item) for item in check_descriptors))
                np.testing.assert_allclose(check_arrays[-1], expected, rtol=2e-4, atol=2e-5)
                abs_errors.append(float(np.max(np.abs(check_arrays[-1] - expected))))
            records[profile]["max_abs_error"] = max(abs_errors)
            records[profile]["correctness"] = "PASS"
        # All compilation/reference work completes before timing any profile.
        for profile, function in kernels.items():
            count = 1
            for _ in range(8):
                elapsed = shim.run_batch(function, *pointers, count)
                if elapsed >= args.batch_ms * 0.5:
                    break
                count = min(1000000, max(count + 1, math.ceil(count * args.batch_ms / max(elapsed, 0.001))))
            iters[profile] = count
            for _ in range(5):
                shim.run_batch(function, *pointers, count)
            samples[profile] = []
        order_rng = random.Random(20261105)
        for _ in range(args.rounds):
            order = list(kernels)
            order_rng.shuffle(order)
            for profile in order:
                elapsed = shim.run_batch(kernels[profile], *pointers, iters[profile])
                samples[profile].append(elapsed / iters[profile])
        for profile, values in samples.items():
            records[profile].update(summarize(values))
            records[profile]["iterations_per_batch"] = iters[profile]
            records[profile]["output_elements_per_second"] = math.prod(case["yshape"]) / (records[profile]["median_ms"] / 1000)
            arrays[-1].fill(np.nan)
            kernels[profile](*pointers)
            np.testing.assert_allclose(arrays[-1], reference, rtol=2e-4, atol=2e-5)
        baseline = records.get("off", {}).get("median_ms")
        for profile in records:
            if baseline is not None:
                records[profile]["speedup_vs_off"] = baseline / records[profile]["median_ms"]
        report["workloads"][name] = {"input_shape": case["xshape"], "output_shape": case["yshape"], "profiles": records}
        (output / "native-benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(name + ": " + ", ".join(f"{p}={v['median_ms']:.5f} ms" for p, v in records.items()), flush=True)
    print(f"Saved {output / 'native-benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()
