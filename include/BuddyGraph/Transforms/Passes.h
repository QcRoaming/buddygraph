#ifndef BUDDYGRAPH_TRANSFORMS_PASSES_H
#define BUDDYGRAPH_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace buddy::bgraph {

std::unique_ptr<mlir::Pass> createInferShapesPass();
std::unique_ptr<mlir::Pass> createFoldBatchNormIntoConvPass();
std::unique_ptr<mlir::Pass> createFuseElementwisePass();
std::unique_ptr<mlir::Pass> createConvertBGraphToLinalgPass();

#define GEN_PASS_REGISTRATION
#include "BuddyGraph/Transforms/Passes.h.inc"

} // namespace buddy::bgraph

#endif // BUDDYGRAPH_TRANSFORMS_PASSES_H
