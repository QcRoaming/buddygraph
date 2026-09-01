#ifndef BUDDYGRAPH_IR_BGRAP_OPS_H
#define BUDDYGRAPH_IR_BGRAP_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "BuddyGraph/IR/BGraphOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "BuddyGraph/IR/BGraphOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "BuddyGraph/IR/BGraphOps.h.inc"

#endif // BUDDYGRAPH_IR_BGRAP_OPS_H
