add_mlir_dialect_library(BuddyGraphIR
  BGraphDialect.cpp
  BGraphOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/BuddyGraph

  DEPENDS
  MLIRBGraphOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRTransformUtils
)
