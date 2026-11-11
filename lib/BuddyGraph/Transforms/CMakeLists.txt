add_mlir_library(BuddyGraphTransforms
  InferShapes.cpp
  FoldBatchNorm.cpp
  FuseElementwise.cpp
  CPUTransforms.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/BuddyGraph

  DEPENDS
  BuddyGraphPassesIncGen
  MLIRBGraphOpsIncGen

  LINK_LIBS PUBLIC
  BuddyGraphIR
  MLIRArithDialect
  MLIRBufferizationPipelines
  MLIRBufferizationTransforms
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRTransforms
  MLIRVectorDialect
)
