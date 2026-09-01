#include "BuddyGraph/Transforms/Passes.h"

#include "BuddyGraph/IR/BGraphDialect.h"
#include "BuddyGraph/IR/BGraphOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace buddy::bgraph {
#define GEN_PASS_DEF_BGRAPHINFERSHAPES
#include "BuddyGraph/Transforms/Passes.h.inc"

namespace {

FailureOr<int64_t> inferConvSpatialDim(int64_t inputSize, int64_t kernelSize,
                                       int64_t padBefore, int64_t padAfter,
                                       int64_t stride, int64_t dilation) {
  if (ShapedType::isDynamic(inputSize) || ShapedType::isDynamic(kernelSize))
    return ShapedType::kDynamic;
  if (inputSize < 0 || kernelSize <= 0)
    return failure();

  int64_t effectiveKernel;
  if (llvm::MulOverflow(dilation, kernelSize - 1, effectiveKernel) ||
      llvm::AddOverflow(effectiveKernel, int64_t{1}, effectiveKernel))
    return failure();

  int64_t paddedInput;
  if (llvm::AddOverflow(inputSize, padBefore, paddedInput) ||
      llvm::AddOverflow(paddedInput, padAfter, paddedInput) ||
      paddedInput < effectiveKernel)
    return failure();

  int64_t outputSize = (paddedInput - effectiveKernel) / stride;
  if (llvm::AddOverflow(outputSize, int64_t{1}, outputSize) || outputSize <= 0)
    return failure();
  return outputSize;
}

FailureOr<SmallVector<int64_t>> broadcast(ArrayRef<int64_t> lhs,
                                          ArrayRef<int64_t> rhs) {
  size_t rank = std::max(lhs.size(), rhs.size());
  SmallVector<int64_t> result(rank, 1);
  for (size_t offset = 0; offset < rank; ++offset) {
    int64_t left = offset < lhs.size() ? lhs[lhs.size() - 1 - offset] : 1;
    int64_t right = offset < rhs.size() ? rhs[rhs.size() - 1 - offset] : 1;
    int64_t dim;
    if (left == right)
      dim = left;
    else if (left == 1)
      dim = right;
    else if (right == 1)
      dim = left;
    else if (ShapedType::isDynamic(left) || ShapedType::isDynamic(right))
      dim = ShapedType::kDynamic;
    else
      return failure();
    result[rank - 1 - offset] = dim;
  }
  return result;
}

FailureOr<SmallVector<int64_t>> inferShape(Operation *operation) {
  if (auto op = dyn_cast<ReluOp>(operation))
    return SmallVector<int64_t>(
        cast<RankedTensorType>(op.getInput().getType()).getShape());
  if (auto op = dyn_cast<ClampOp>(operation))
    return SmallVector<int64_t>(
        cast<RankedTensorType>(op.getInput().getType()).getShape());
  if (auto op = dyn_cast<BatchNormOp>(operation))
    return SmallVector<int64_t>(
        cast<RankedTensorType>(op.getInput().getType()).getShape());

  auto inferBinary = [](Value lhs,
                        Value rhs) -> FailureOr<SmallVector<int64_t>> {
    return broadcast(cast<RankedTensorType>(lhs.getType()).getShape(),
                     cast<RankedTensorType>(rhs.getType()).getShape());
  };
  if (auto op = dyn_cast<AddOp>(operation))
    return inferBinary(op.getLhs(), op.getRhs());
  if (auto op = dyn_cast<SubOp>(operation))
    return inferBinary(op.getLhs(), op.getRhs());
  if (auto op = dyn_cast<MulOp>(operation))
    return inferBinary(op.getLhs(), op.getRhs());
  if (auto op = dyn_cast<DivOp>(operation))
    return inferBinary(op.getLhs(), op.getRhs());

  if (auto op = dyn_cast<Conv2DOp>(operation)) {
    auto input = cast<RankedTensorType>(op.getInput().getType());
    auto filter = cast<RankedTensorType>(op.getFilter().getType());
    bool nchw = op.getLayout() == Layout::NCHW;
    SmallVector<int64_t> shape(input.getShape());
    shape[nchw ? 1 : 3] = filter.getDimSize(0);
    ArrayRef<int64_t> pads = op.getPads();
    ArrayRef<int64_t> strides = op.getStrides();
    ArrayRef<int64_t> dilations = op.getDilations();
    auto height = inferConvSpatialDim(input.getDimSize(nchw ? 2 : 1),
                                      filter.getDimSize(nchw ? 2 : 1), pads[0],
                                      pads[2], strides[0], dilations[0]);
    auto width = inferConvSpatialDim(input.getDimSize(nchw ? 3 : 2),
                                     filter.getDimSize(nchw ? 3 : 2), pads[1],
                                     pads[3], strides[1], dilations[1]);
    if (failed(height) || failed(width))
      return failure();
    shape[nchw ? 2 : 1] = *height;
    shape[nchw ? 3 : 2] = *width;
    return shape;
  }

  if (auto op = dyn_cast<ReshapeOp>(operation)) {
    auto input = cast<RankedTensorType>(op.getInput().getType());
    SmallVector<int64_t> shape(op.getShape().begin(), op.getShape().end());
    int64_t inferIndex = -1;
    int64_t known = 1;
    for (auto [index, dim] : llvm::enumerate(shape)) {
      if (dim == -1)
        inferIndex = index;
      else
        known *= dim;
    }
    if (inferIndex >= 0 && input.hasStaticShape())
      shape[inferIndex] = input.getNumElements() / known;
    else if (inferIndex >= 0)
      shape[inferIndex] = ShapedType::kDynamic;
    return shape;
  }

  if (auto op = dyn_cast<TransposeOp>(operation)) {
    auto input = cast<RankedTensorType>(op.getInput().getType());
    SmallVector<int64_t> shape;
    for (int64_t axis : op.getPermutation())
      shape.push_back(input.getDimSize(axis));
    return shape;
  }

  if (auto op = dyn_cast<ReduceMeanOp>(operation)) {
    auto input = cast<RankedTensorType>(op.getInput().getType());
    llvm::SmallBitVector reduced(input.getRank());
    for (int64_t axis : op.getAxes())
      reduced.set(axis < 0 ? axis + input.getRank() : axis);
    SmallVector<int64_t> shape;
    for (int64_t index = 0; index < input.getRank(); ++index) {
      if (!reduced[index])
        shape.push_back(input.getDimSize(index));
      else if (op.getKeepDims())
        shape.push_back(1);
    }
    return shape;
  }

  if (auto op = dyn_cast<FusedElementwiseOp>(operation)) {
    SmallVector<int64_t> shape(
        cast<RankedTensorType>(op.getInputs().front().getType()).getShape());
    for (Value input : op.getInputs().drop_front()) {
      auto next =
          broadcast(shape, cast<RankedTensorType>(input.getType()).getShape());
      if (failed(next))
        return failure();
      shape = std::move(*next);
    }
    return shape;
  }

  return failure();
}

class BGraphInferShapes
    : public impl::BGraphInferShapesBase<BGraphInferShapes> {
public:
  using Base::Base;

  void runOnOperation() final {
    WalkResult result = getOperation().walk([&](Operation *operation) {
      if (operation->getName().getDialectNamespace() != "bgraph" ||
          isa<YieldOp>(operation))
        return WalkResult::advance();
      auto shape = inferShape(operation);
      if (failed(shape)) {
        operation->emitError("could not infer a legal result shape");
        return WalkResult::interrupt();
      }
      auto oldType = cast<RankedTensorType>(operation->getResult(0).getType());
      operation->getResult(0).setType(
          RankedTensorType::get(*shape, oldType.getElementType()));
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createInferShapesPass() {
  return std::make_unique<BGraphInferShapes>();
}

} // namespace buddy::bgraph
