# Source from Bash: source scripts/env.sh
# Override these paths before sourcing to use another precompiled MLIR SDK.
export BUDDYGRAPH_CACHE="${BUDDYGRAPH_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/buddygraph}"
export LLVM_BUILD="${LLVM_BUILD:-$BUDDYGRAPH_CACHE/toolchain/usr/lib/llvm-21}"
export BUDDYGRAPH_MLIR_PYTHON_DIR="${BUDDYGRAPH_MLIR_PYTHON_DIR:-$BUDDYGRAPH_CACHE/python-build/python_packages/mlir_core}"
export PATH="$BUDDYGRAPH_CACHE/venv/bin:$LLVM_BUILD/bin:$PATH"
export PYTHONPATH="$BUDDYGRAPH_MLIR_PYTHON_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$LLVM_BUILD/lib:$BUDDYGRAPH_CACHE/toolchain/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
