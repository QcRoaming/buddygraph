// RUN: buddygraph-opt --bgraph-fuse-elementwise --convert-bgraph-to-linalg --one-shot-bufferize="bufferize-function-boundaries" %S/../../Dialect/BGraph/fuse-elementwise.mlir | FileCheck %s

// CHECK-NOT: bgraph.
// CHECK-NOT: bufferization.to_
// CHECK: memref.alloc
// CHECK: linalg.generic
