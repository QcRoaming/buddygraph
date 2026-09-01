add_mlir_library(BGraphToLinalg
  BGraphToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/BuddyGraph

  DEPENDS
  BuddyGraphPassesIncGen
  MLIRBGraphOpsIncGen

  LINK_LIBS PUBLIC
  BuddyGraphIR
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransformUtils
  MLIRTransforms
)
