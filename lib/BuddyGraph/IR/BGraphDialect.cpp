#include "BuddyGraph/IR/BGraphDialect.h"
#include "BuddyGraph/IR/BGraphOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace buddy::bgraph;

#include "BuddyGraph/IR/BGraphOpsDialect.cpp.inc"

void BGraphDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "BuddyGraph/IR/BGraphOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "BuddyGraph/IR/BGraphOpsAttributes.cpp.inc"
      >();
}

#include "BuddyGraph/IR/BGraphOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "BuddyGraph/IR/BGraphOpsAttributes.cpp.inc"
