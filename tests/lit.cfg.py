import os

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "BUDDYGRAPH"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir", ".py"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.buddygraph_obj_root, "tests")
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py"]

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.environment["PYTHONPATH"] = os.getenv("PYTHONPATH", "")
llvm_config.with_environment(
    "PYTHONPATH",
    [
        os.path.join(config.buddygraph_source_root, ".deps"),
        os.path.join(config.llvm_build_root, "tools", "mlir", "python_packages", "mlir_core"),
    ],
    append_path=True,
)

buddygraph_tools_dir = os.path.join(config.buddygraph_obj_root, "bin")
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.add_tool_substitutions(
    [
        "buddygraph-opt",
        "FileCheck",
        "count",
        "not",
        ToolSubst("%PYTHON", config.python_executable, unresolved="fatal"),
    ],
    [buddygraph_tools_dir, config.llvm_tools_dir],
)
config.substitutions.extend(
    [
        ("%buddygraph_src_root", config.buddygraph_source_root),
        ("%buddygraph_obj_root", config.buddygraph_obj_root),
        ("%llvm_tools_dir", config.llvm_tools_dir),
    ]
)
