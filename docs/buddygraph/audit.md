# BuddyGraph Phase 0 Audit

Audit time: 2026-07-28T09:09:30Z

## Outcome

The checked-in LLVM/MLIR and Buddy-MLIR builds are usable.  A minimal
`func`/`arith` module verifies with both `build/bin/buddy-opt` and
`llvm/build/bin/mlir-opt`, and the existing `buddy-opt` target builds
successfully.  BuddyGraph will therefore reuse the existing LLVM/MLIR build
and will not rebuild LLVM.

The repository has substantial user work in progress, including changes in
the exact files that an in-tree dialect would normally edit.  To protect that
work, BuddyGraph is implemented as an out-of-tree CMake project rooted at
`jlq/projects/buddygraph`.  It provides a dedicated `buddygraph-opt` driver
with the same MLIR dialect/pass discovery and pass-pipeline behavior.  No
existing Buddy dialect, Transform/Microkernel source, or root build file is
modified.

## Revisions and build environment

- Buddy-MLIR root: `/buddy-mlir`
- Buddy-MLIR commit: `d7bb40cbac731175dc507f06f0655d81508f2ca2`
- LLVM submodule commit:
  `09b849a2ac83dbf4be1b2f01c767339e46cc34ae`
- Reported LLVM/MLIR version: `21.0.0git`, optimized build with assertions
- Buddy build: `/buddy-mlir/build`, `CMAKE_BUILD_TYPE=RELEASE`
- LLVM build: `/buddy-mlir/llvm/build`, `CMAKE_BUILD_TYPE=Release`
- MLIR package: `/buddy-mlir/llvm/build/lib/cmake/mlir`
- LLVM package: `/buddy-mlir/llvm/build/lib/cmake/llvm`
- Compiler recorded by the Buddy build: `/usr/bin/c++`
- Existing reusable targets include `buddy-opt` and `check-buddy`.
- LLVM tools available for this project include `mlir-opt`, `llvm-lit`, and
  `FileCheck`.

`git submodule status` reports the LLVM revision but then fails with
`no submodule mapping found in .gitmodules for path
'riscv-ime-extension-spec'`.  This pre-existing repository metadata issue does
not affect the LLVM checkout or the isolated BuddyGraph build.

## Repository conventions

The in-tree Buddy layout uses:

- dialect ODS: `midend/include/Dialect/<Name>`;
- dialect implementation: `midend/lib/Dialect/<Name>`;
- conversions: `midend/lib/Conversion/<Name>`;
- driver registration: `tools/buddy-opt/buddy-opt.cpp`;
- regression tests: `tests`, configured by `tests/lit.cfg.py` and
  `tests/lit.site.cfg.py.in`;
- generated ODS targets: `mlir_tablegen`, `add_mlir_dialect`, and
  `add_mlir_dialect_library`.

The isolated project mirrors those responsibilities under `include/`, `lib/`,
`tools/`, `frontend/`, and `tests/`.  Generated TableGen headers go into the
project build tree.  The planned targets are:

- `MLIRBGraphOpsIncGen` and `BuddyGraphPassesIncGen` for generated declarations;
- `BuddyGraphIR` for the dialect and operation implementation;
- `BuddyGraphTransforms` for custom rewrites and passes;
- `BGraphToLinalg` for the full conversion;
- `buddygraph-opt` for the registered optimizer driver;
- `check-buddygraph` for the isolated lit suite.

## Registration and test entry points

The existing `buddy-opt` registers all upstream MLIR passes, inserts Buddy
dialects in a `DialectRegistry`, explicitly registers Buddy passes, and calls
`MlirOptMain`.  `buddygraph-opt` will follow this mechanism while registering
`buddy::bgraph::BGraphDialect` and the four required `bgraph-*`/conversion
passes.

The root test suite uses `llvm-lit` with `FileCheck`, `count`, and `not` tool
substitutions.  BuddyGraph will use a local lit configuration and the existing
LLVM tool directory so that tests remain runnable without touching
`check-buddy`.

Baseline commands executed successfully:

```bash
./build/bin/buddy-opt --version
ninja -C build buddy-opt
printf '<minimal module>' | ./build/bin/buddy-opt --verify-each -o /dev/null -
printf '<minimal module>' | ./llvm/build/bin/mlir-opt --verify-each -o /dev/null -
./llvm/build/bin/llvm-lit --version
./llvm/build/bin/FileCheck --version
```

## Python and frontend status

- The LLVM build has `MLIR_ENABLE_BINDINGS_PYTHON=ON`.
- The Buddy build has `BUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- With
  `PYTHONPATH=/buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/buddy-mlir/build/python_packages`,
  both `mlir.ir` and `buddy` import successfully.
- `numpy` is installed in system Python and the existing
  `/buddy-mlir/jlq/skills/.venv`.
- `onnx` and `onnxruntime` are not currently installed in either checked
  environment.

The formal importer will use the MLIR Python API (`Operation.create` and
builtin/func/arith operations), not handwritten MLIR text.  ONNX is a necessary
small frontend dependency and will be listed explicitly.  NumPy is sufficient
for the deterministic reference implementation, so ONNX Runtime remains
optional.

## Protected local work

The root worktree contains modified and untracked research files.  Particularly
sensitive overlapping files include:

- `tools/buddy-opt/buddy-opt.cpp` and `tools/buddy-opt/CMakeLists.txt`;
- `midend/include/Dialect/CMakeLists.txt` and
  `midend/lib/Dialect/CMakeLists.txt`;
- `midend/lib/Conversion/MatMulOptimization/*`;
- the untracked `midend/include/Dialect/Microkernel` and
  `midend/lib/Dialect/Microkernel` trees;
- many files under `jlq/thesis`.

The four registration/CMake files already contain Microkernel-related user
changes.  BuddyGraph will not edit them.  It will also not stage, stash,
delete, reset, commit, or push any repository content.

## Local API risks and decisions

1. The local API is MLIR 21 development head, so the implementation will be
   compiled against local headers rather than copied from an older online
   example.
2. `buddygraph-opt`, rather than the root `buddy-opt`, is the registration
   surface because the root registration files have protected edits.  This is
   the only deliberate deviation from the guide's in-tree target name.
3. The existing Buddy Python package has no generated BGraph bindings.  The
   importer will construct registered generic BGraph operations with the MLIR
   Python API and rely on `buddygraph-opt` for verification and lowering.
4. ONNX and ONNX Runtime are absent.  The core hand-written MLIR pipeline will
   be built first; ONNX will be installed or otherwise made available only
   when the frontend stage is reached.
5. Static f32 tensors are the MVP.  Dynamic batch support is not allowed to
   weaken the static end-to-end closure.

## Implementation plan

1. Define the `bgraph` dialect, layout enum attribute, required operations,
   verifiers, folds, and canonicalizations in ODS plus focused C++.
2. Add shape inference, BatchNorm-into-Conv folding, and single-use
   elementwise-chain fusion into a region-based `bgraph.fused_elementwise`.
3. Implement `applyFullConversion` lowering for every BGraph operation to
   Linalg/Tensor/Arith/Math and prove that no `bgraph.*` operation remains.
4. Add a fixed-opset ONNX importer and deterministic model generator using
   the MLIR Python API.
5. Add lit verifier/rewrite/conversion tests, deterministic numerical E2E
   tests, metric collection, and the required architecture/debugging/results
   documentation.

The initial build command is:

```bash
cmake -S /buddy-mlir/jlq/projects/buddygraph \
  -B /buddy-mlir/jlq/projects/buddygraph/build -G Ninja \
  -DMLIR_DIR=/buddy-mlir/llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=/buddy-mlir/llvm/build/lib/cmake/llvm
cmake --build /buddy-mlir/jlq/projects/buddygraph/build --target buddygraph-opt
```
