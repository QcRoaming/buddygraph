add_mlir_library(BuddyGraphTransforms
  InferShapes.cpp
  FoldBatchNorm.cpp
  FuseElementwise.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/BuddyGraph

  DEPENDS
  BuddyGraphPassesIncGen
  MLIRBGraphOpsIncGen

  LINK_LIBS PUBLIC
  BuddyGraphIR
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
