// RUN: %PYTHON %buddygraph_src_root/frontend/BGraph/generate_test_models.py %t.models
// RUN: %PYTHON %buddygraph_src_root/frontend/BGraph/import_onnx.py %t.models/conv_bn_relu.onnx -o %t.conv.mlir
// RUN: buddygraph-opt -mlir-print-debuginfo %t.conv.mlir | FileCheck %s --check-prefix=CONV
// RUN: %PYTHON %buddygraph_src_root/frontend/BGraph/import_onnx.py %t.models/elementwise_chain.onnx -o %t.elt.mlir
// RUN: buddygraph-opt -mlir-print-debuginfo %t.elt.mlir | FileCheck %s --check-prefix=ELT
// RUN: %PYTHON %buddygraph_src_root/frontend/BGraph/import_onnx.py %t.models/clamp_constant.onnx -o %t.clamp.mlir
// RUN: buddygraph-opt -mlir-print-debuginfo %t.clamp.mlir | FileCheck %s --check-prefix=CLAMP
// RUN: %PYTHON %buddygraph_src_root/frontend/BGraph/import_onnx.py %t.models/clamp_shared_initializer.onnx -o %t.clamp-shared.mlir
// RUN: buddygraph-opt -mlir-print-debuginfo %t.clamp-shared.mlir | FileCheck %s --check-prefix=CLAMP-SHARED
// RUN: not %PYTHON %buddygraph_src_root/frontend/BGraph/import_onnx.py %t.models/unsupported_operator.onnx 2>&1 | FileCheck %s --check-prefix=OP
// RUN: not %PYTHON %buddygraph_src_root/frontend/BGraph/import_onnx.py %t.models/wrong_opset.onnx 2>&1 | FileCheck %s --check-prefix=OPSET
// RUN: not %PYTHON %buddygraph_src_root/frontend/BGraph/import_onnx.py %t.models/unsupported_attribute.onnx 2>&1 | FileCheck %s --check-prefix=ATTR
// RUN: not %PYTHON %buddygraph_src_root/frontend/BGraph/import_onnx.py %t.models/clip_runtime_bound.onnx 2>&1 | FileCheck %s --check-prefix=CLAMP-RUNTIME
// RUN: not %PYTHON %buddygraph_src_root/frontend/BGraph/import_onnx.py %t.models/clip_missing_bound.onnx 2>&1 | FileCheck %s --check-prefix=CLAMP-MISSING
// RUN: not %PYTHON %buddygraph_src_root/frontend/BGraph/import_onnx.py %t.models/clip_nonscalar_bound.onnx 2>&1 | FileCheck %s --check-prefix=CLAMP-SHAPE
// RUN: not %PYTHON %buddygraph_src_root/frontend/BGraph/import_onnx.py %t.models/clip_nonfinite_bound.onnx 2>&1 | FileCheck %s --check-prefix=CLAMP-FINITE
// RUN: not %PYTHON %buddygraph_src_root/frontend/BGraph/import_onnx.py %t.models/clip_reversed_bound.onnx 2>&1 | FileCheck %s --check-prefix=CLAMP-ORDER

// CONV-LABEL: func.func public @main_graph
// CONV: "bgraph.conv2d"
// CONV-SAME: layout = #bgraph.layout<nchw>
// CONV: "bgraph.batch_norm"
// CONV: "bgraph.relu"
// CONV-DAG: loc("initializer:w")
// CONV-DAG: loc("conv")
// CONV-DAG: loc("batch_norm")
// CONV-DAG: loc("relu")

// ELT-LABEL: func.func public @main_graph
// ELT: "bgraph.add"
// ELT: "bgraph.sub"
// ELT: "bgraph.div"
// ELT: "bgraph.relu"
// ELT: "bgraph.reshape"
// ELT-SAME: shape = array<i64: 3, 2>
// ELT: "bgraph.transpose"
// ELT-SAME: permutation = array<i64: 1, 0>
// ELT-DAG: loc("broadcast_add")
// ELT-DAG: loc("broadcast_sub")
// ELT-DAG: loc("broadcast_div")
// ELT-DAG: loc("relu")
// ELT-DAG: loc("reshape")
// ELT-DAG: loc("transpose")

// CLAMP-LABEL: func.func public @main_graph
// CLAMP: "bgraph.add"
// CLAMP: "bgraph.clamp"
// CLAMP-SAME: max_value = 1.000000e+00 : f32
// CLAMP-SAME: min_value = -1.000000e+00 : f32
// CLAMP: "bgraph.mul"
// CLAMP: "bgraph.reduce_mean"
// CLAMP-DAG: loc("clip")

// CLAMP-SHARED-LABEL: func.func public @main_graph
// CLAMP-SHARED: %[[BOUND:.*]] = arith.constant dense<1.000000e+00>
// CLAMP-SHARED: %[[SUM:.*]] = "bgraph.add"({{.*}}, %[[BOUND]])
// CLAMP-SHARED: "bgraph.clamp"(%[[SUM]])

// OP: buddygraph-import-onnx: error: node 'unsupported_softmax' (Softmax): unsupported operator
// OPSET: buddygraph-import-onnx: error: expected exactly ONNX opset 18
// ATTR: buddygraph-import-onnx: error: node 'auto_pad_conv' (Conv): auto_pad='SAME_UPPER' is unsupported
// CLAMP-RUNTIME: buddygraph-import-onnx: error: node 'runtime_clip_bound' (Clip): 'runtime_min' must be a constant initializer
// CLAMP-MISSING: buddygraph-import-onnx: error: node 'missing_clip_bound' (Clip): expected data, constant scalar min, and constant scalar max inputs
// CLAMP-SHAPE: buddygraph-import-onnx: error: node 'nonscalar_clip_bound' (Clip): constant 'clip_min' must be a scalar
// CLAMP-FINITE: buddygraph-import-onnx: error: node 'nonfinite_clip_bound' (Clip): constant 'clip_min' must be finite
// CLAMP-ORDER: buddygraph-import-onnx: error: node 'reversed_clip_bound' (Clip): min bound must be less than or equal to max bound
