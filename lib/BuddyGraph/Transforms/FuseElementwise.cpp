#include "BuddyGraph/Transforms/Passes.h"

#include "BuddyGraph/IR/BGraphDialect.h"
#include "BuddyGraph/IR/BGraphOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace buddy::bgraph {
#define GEN_PASS_DEF_BGRAPHFUSEELEMENTWISE
#include "BuddyGraph/Transforms/Passes.h.inc"

namespace {

bool isFusible(Operation *operation) {
  return isa<AddOp, SubOp, MulOp, DivOp, ReluOp, ClampOp>(operation) &&
         isMemoryEffectFree(operation);
}

struct ElementwiseTree {
  llvm::SmallPtrSet<Operation *, 8> operations;
  SmallVector<Operation *> orderedOperations;
  llvm::SetVector<Value> leaves;

  void collect(Operation *operation) {
    if (!operations.insert(operation).second)
      return;
    orderedOperations.push_back(operation);
    for (Value operand : operation->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      if (producer && isFusible(producer) && operand.hasOneUse())
        collect(producer);
      else
        leaves.insert(operand);
    }
  }
};

FailureOr<Value> emitScalar(Operation *operation, const ElementwiseTree &tree,
                            IRMapping &mapping, PatternRewriter &rewriter) {
  auto scalarFor = [&](Value operand) -> FailureOr<Value> {
    Operation *producer = operand.getDefiningOp();
    if (producer && tree.operations.contains(producer))
      return emitScalar(producer, tree, mapping, rewriter);
    Value mapped = mapping.lookupOrNull(operand);
    if (mapped)
      return mapped;
    return failure();
  };

  if (auto relu = dyn_cast<ReluOp>(operation)) {
    auto input = scalarFor(relu.getInput());
    if (failed(input))
      return failure();
    auto zero = rewriter.create<arith::ConstantFloatOp>(
        operation->getLoc(), APFloat(0.0f), rewriter.getF32Type());
    return rewriter.create<arith::MaximumFOp>(operation->getLoc(), *input, zero)
        .getResult();
  }

  if (auto clamp = dyn_cast<ClampOp>(operation)) {
    auto input = scalarFor(clamp.getInput());
    if (failed(input))
      return failure();
    Value minimum = rewriter.create<arith::ConstantFloatOp>(
        operation->getLoc(), clamp.getMinValue(), rewriter.getF32Type());
    Value maximum = rewriter.create<arith::ConstantFloatOp>(
        operation->getLoc(), clamp.getMaxValue(), rewriter.getF32Type());
    Value below = rewriter.create<arith::CmpFOp>(
        operation->getLoc(), arith::CmpFPredicate::OLT, *input, minimum);
    Value lowerClamped = rewriter.create<arith::SelectOp>(
        operation->getLoc(), below, minimum, *input);
    Value above = rewriter.create<arith::CmpFOp>(
        operation->getLoc(), arith::CmpFPredicate::OLT, maximum, lowerClamped);
    return rewriter
        .create<arith::SelectOp>(operation->getLoc(), above, maximum,
                                 lowerClamped)
        .getResult();
  }

  auto lhs = scalarFor(operation->getOperand(0));
  auto rhs = scalarFor(operation->getOperand(1));
  if (failed(lhs) || failed(rhs))
    return failure();
  if (isa<AddOp>(operation))
    return rewriter.create<arith::AddFOp>(operation->getLoc(), *lhs, *rhs)
        .getResult();
  if (isa<SubOp>(operation))
    return rewriter.create<arith::SubFOp>(operation->getLoc(), *lhs, *rhs)
        .getResult();
  if (isa<MulOp>(operation))
    return rewriter.create<arith::MulFOp>(operation->getLoc(), *lhs, *rhs)
        .getResult();
  if (isa<DivOp>(operation))
    return rewriter.create<arith::DivFOp>(operation->getLoc(), *lhs, *rhs)
        .getResult();
  return failure();
}

template <typename RootOp>
class FuseElementwisePattern final : public OpRewritePattern<RootOp> {
public:
  using OpRewritePattern<RootOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RootOp root,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(root->getUsers(),
                     [](Operation *user) { return isFusible(user); }))
      return rewriter.notifyMatchFailure(root, "operation is not a chain root");
    ElementwiseTree tree;
    tree.collect(root);
    if (tree.operations.size() < 2)
      return rewriter.notifyMatchFailure(root,
                                         "chain has no removable intermediate");

    SmallVector<Location> locations;
    locations.reserve(tree.orderedOperations.size());
    for (Operation *operation : tree.orderedOperations)
      locations.push_back(operation->getLoc());
    Location location = rewriter.getFusedLoc(locations);

    OperationState state(location, FusedElementwiseOp::getOperationName());
    state.addOperands(tree.leaves.getArrayRef());
    state.addTypes(root.getResult().getType());
    state.addRegion();
    auto fused = cast<FusedElementwiseOp>(rewriter.create(state));

    Region &body = fused.getBody();
    Block *block = rewriter.createBlock(&body);
    IRMapping mapping;
    for (Value leaf : tree.leaves) {
      Type elementType =
          cast<RankedTensorType>(leaf.getType()).getElementType();
      BlockArgument argument = block->addArgument(elementType, location);
      mapping.map(leaf, argument);
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);
    auto scalar = emitScalar(root, tree, mapping, rewriter);
    if (failed(scalar)) {
      rewriter.eraseOp(fused);
      return rewriter.notifyMatchFailure(root, "could not build scalar body");
    }
    rewriter.create<YieldOp>(location, *scalar);

    rewriter.replaceOp(root, fused.getResult());
    for (Operation *operation : tree.orderedOperations)
      if (operation != root && operation->use_empty())
        rewriter.eraseOp(operation);
    return success();
  }
};

class BGraphFuseElementwise
    : public impl::BGraphFuseElementwiseBase<BGraphFuseElementwise> {
public:
  using Base::Base;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns
        .add<FuseElementwisePattern<AddOp>, FuseElementwisePattern<SubOp>,
             FuseElementwisePattern<MulOp>, FuseElementwisePattern<DivOp>,
             FuseElementwisePattern<ReluOp>, FuseElementwisePattern<ClampOp>>(
            &getContext());
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFuseElementwisePass() {
  return std::make_unique<BGraphFuseElementwise>();
}

} // namespace buddy::bgraph
