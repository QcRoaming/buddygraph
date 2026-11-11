# RUN: %PYTHON %s %buddygraph_src_root %buddygraph_obj_root %llvm_tools_dir %t | FileCheck %s
# CHECK: direct_output=verified
# CHECK: conv_tails=verified dilation=verified stride_fallback=verified
# CHECK: alternate_schedule=verified
# CHECK: status=PASS

"""Check complete native outputs after destination propagation and CPU scheduling."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    source, build, llvm, temporary = map(Path, sys.argv[1:])
    sys.path.insert(0, str(source / "scripts"))
    from benchmark_native import compile_kernel, descriptor, inputs_and_reference, workload

    optimizer = build / "bin/buddygraph-opt"
    cases = [
        ("relu", workload("relu", 17)),
        ("clamp", workload("clamp", 17)),
        ("tails", workload("conv_bn", 11, in_channels=3, out_channels=5)),
        ("small", workload("conv_bn", 3, in_channels=2, out_channels=3, kernel=1)),
        ("dilated", workload("conv_bn", 9, in_channels=3, out_channels=7, dilation=2)),
        ("stride", workload("conv_bn", 11, in_channels=3, out_channels=5, stride=2)),
    ]
    for name, case in cases:
        for profile in ("all", "direct_output", "cpu"):
            library, metrics = compile_kernel(
                case, profile, temporary / name / profile, optimizer, llvm, "gcc"
            )
            function = library._mlir_ciface_kernel
            function.argtypes = [ctypes.c_void_p] * 4
            function.restype = None
            for seed in (11, 22, 33):
                arrays, expected = inputs_and_reference(case, seed)
                arrays[-1].fill(np.nan)
                descriptions = [descriptor(array) for array in arrays]
                function(*(ctypes.byref(item) for item in descriptions))
                np.testing.assert_allclose(arrays[-1], expected, rtol=2e-4, atol=2e-5)
            if profile != "all" and name in ("relu", "clamp"):
                assert metrics["memref_alloc_ops"] == 0
                assert metrics["memref_copy_ops"] == 0
            if profile == "cpu" and name in ("tails", "dilated"):
                assert metrics["scheduled_vector_load_ops"] > 0
                assert metrics["scheduled_linalg_conv_ops"] == 0
            if profile == "cpu" and name == "stride":
                assert metrics["scheduled_linalg_conv_ops"] == 1

    # A second schedule must preserve complete outputs with different tails.
    case = workload("conv_bn", 11, in_channels=3, out_channels=5)
    library, metrics = compile_kernel(
        case, "cpu", temporary / "alternate", optimizer, llvm, "gcc",
        vector_width=4, channel_tile=2,
    )
    function = library._mlir_ciface_kernel
    function.argtypes = [ctypes.c_void_p] * 4
    function.restype = None
    arrays, expected = inputs_and_reference(case, 44)
    arrays[-1].fill(np.nan)
    descriptions = [descriptor(array) for array in arrays]
    function(*(ctypes.byref(item) for item in descriptions))
    np.testing.assert_allclose(arrays[-1], expected, rtol=2e-4, atol=2e-5)
    assert metrics["scheduled_vector_load_ops"] > 0
    assert metrics["scheduled_linalg_conv_ops"] == 0

    print("direct_output=verified")
    print("conv_tails=verified dilation=verified stride_fallback=verified")
    print("alternate_schedule=verified")
    print("status=PASS")


if __name__ == "__main__":
    main()
