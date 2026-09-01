#include "BuddyGraph/IR/BGraphOps.h"

#include "BuddyGraph/IR/BGraphDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/MathExtras.h"

#include <cmath>
#include <limits>

using namespace mlir;
using namespace buddy::bgraph;

namespace {

FailureOr<RankedTensorType> getF32Tensor(Operation *op, Type type,
                                         StringRef role) {
  if (!type)
    return op->emitOpError() << role << " has no type";
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return op->emitOpError() << role << " must be a ranked tensor";
  if (!tensor.getElementType().isF32())
    return op->emitOpError() << role << " must have f32 elements";
  return tensor;
}

bool areCompatibleDims(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}

bool areCompatibleShapes(ArrayRef<int64_t> expected, ArrayRef<int64_t> actual) {
  if (expected.size() != actual.size())
    return false;
  return llvm::all_of(llvm::zip(expected, actual), [](auto dims) {
    return areCompatibleDims(std::get<0>(dims), std::get<1>(dims));
  });
}

FailureOr<SmallVector<int64_t>> inferBroadcastShape(ArrayRef<int64_t> lhs,
                                                    ArrayRef<int64_t> rhs) {
  const size_t rank = std::max(lhs.size(), rhs.size());
  SmallVector<int64_t> result(rank, 1);
  for (size_t offset = 0; offset < rank; ++offset) {
    int64_t lhsDim = offset < lhs.size() ? lhs[lhs.size() - 1 - offset] : 1;
    int64_t rhsDim = offset < rhs.size() ? rhs[rhs.size() - 1 - offset] : 1;
    int64_t inferred = ShapedType::kDynamic;
    if (lhsDim == rhsDim)
      inferred = lhsDim;
    else if (lhsDim == 1)
      inferred = rhsDim;
    else if (rhsDim == 1)
      inferred = lhsDim;
    else if (ShapedType::isDynamic(lhsDim))
      inferred = rhsDim == 1 ? ShapedType::kDynamic : rhsDim;
    else if (ShapedType::isDynamic(rhsDim))
      inferred = lhsDim == 1 ? ShapedType::kDynamic : lhsDim;
    else
      return failure();
    result[rank - 1 - offset] = inferred;
  }
  return result;
}

template <typename OpTy> LogicalResult verifyBinary(OpTy op) {
  auto lhs = getF32Tensor(op, op.getLhs().getType(), "lhs");
  auto rhs = getF32Tensor(op, op.getRhs().getType(), "rhs");
  auto result = getF32Tensor(op, op.getResult().getType(), "result");
  if (failed(lhs) || failed(rhs) || failed(result))
    return failure();
  auto shape = inferBroadcastShape((*lhs).getShape(), (*rhs).getShape());
  if (failed(shape))
    return op.emitOpError("operands are not broadcast compatible");
  if (!areCompatibleShapes(*shape, (*result).getShape()))
    return op.emitOpError() << "result shape " << (*result).getShape()
                            << " does not match broadcast shape " << *shape;
  return success();
}

FailureOr<int64_t> checkedProduct(ArrayRef<int64_t> shape) {
  int64_t value = 1;
  for (int64_t dim : shape) {
    if (ShapedType::isDynamic(dim))
      return failure();
    if (dim < 0 || llvm::MulOverflow(value, dim, value))
      return failure();
  }
  return value;
}

FailureOr<SmallVector<int64_t>> inferReshapeShape(RankedTensorType input,
                                                  ArrayRef<int64_t> target) {
  SmallVector<int64_t> inferred(target.begin(), target.end());
  int64_t inferIndex = -1;
  int64_t knownElements = 1;
  for (auto [index, dim] : llvm::enumerate(target)) {
    if (dim == -1) {
      if (inferIndex != -1)
        return failure();
      inferIndex = index;
      continue;
    }
    if (dim <= 0 || llvm::MulOverflow(knownElements, dim, knownElements))
      return failure();
  }

  auto inputElements = checkedProduct(input.getShape());
  if (inferIndex != -1) {
    if (failed(inputElements) || knownElements == 0 ||
        *inputElements % knownElements != 0)
      inferred[inferIndex] = ShapedType::kDynamic;
    else
      inferred[inferIndex] = *inputElements / knownElements;
  } else if (succeeded(inputElements) && *inputElements != knownElements) {
    return failure();
  }
  return inferred;
}

FailureOr<SmallVector<int64_t>> inferReduceShape(RankedTensorType input,
                                                 ArrayRef<int64_t> axes,
                                                 bool keepDims) {
  llvm::SmallBitVector reduced(input.getRank());
  for (int64_t axis : axes) {
    int64_t normalized = axis < 0 ? axis + input.getRank() : axis;
    if (normalized < 0 || normalized >= input.getRank() || reduced[normalized])
      return failure();
    reduced.set(normalized);
  }
  SmallVector<int64_t> result;
  for (int64_t index = 0; index < input.getRank(); ++index) {
    if (!reduced[index])
      result.push_back(input.getDimSize(index));
    else if (keepDims)
      result.push_back(1);
  }
  return result;
}

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

struct ElideMulByOne final : OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    auto isOneSplat = [](Value value) {
      Attribute attribute;
      if (!matchPattern(value, m_Constant(&attribute)))
        return false;
      if (auto dense = dyn_cast<DenseFPElementsAttr>(attribute))
        return dense.isSplat() &&
               dense.getSplatValue<APFloat>().convertToFloat() == 1.0f;
      if (auto scalar = dyn_cast<FloatAttr>(attribute))
        return scalar.getValueAsDouble() == 1.0;
      return false;
    };
    Value identity;
    if (isOneSplat(op.getRhs()))
      identity = op.getLhs();
    else if (isOneSplat(op.getLhs()))
      identity = op.getRhs();
    if (!identity || identity.getType() != op.getResult().getType())
      return failure();
    rewriter.replaceOp(op, identity);
    return success();
  }
};

struct ElideNestedRelu final : OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    auto inner = op.getInput().getDefiningOp<ReluOp>();
    if (!inner)
      return failure();
    rewriter.replaceOp(op, inner.getResult());
    return success();
  }
};

struct ElideNestedClamp final : OpRewritePattern<ClampOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ClampOp op,
                                PatternRewriter &rewriter) const override {
    auto inner = op.getInput().getDefiningOp<ClampOp>();
    if (!inner || inner.getMinValueAttr() != op.getMinValueAttr() ||
        inner.getMaxValueAttr() != op.getMaxValueAttr() ||
        inner.getResult().getType() != op.getResult().getType())
      return failure();
    rewriter.replaceOp(op, inner.getResult());
    return success();
  }
};

struct CollapseReshape final : OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto inner = op.getInput().getDefiningOp<ReshapeOp>();
    if (!inner)
      return failure();
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getResult().getType(),
                                           inner.getInput(), op.getShapeAttr());
    return success();
  }
};

struct CancelTranspose final : OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto inner = op.getInput().getDefiningOp<TransposeOp>();
    if (!inner || inner.getInput().getType() != op.getResult().getType())
      return failure();
    ArrayRef<int64_t> first = inner.getPermutation();
    ArrayRef<int64_t> second = op.getPermutation();
    if (first.size() != second.size())
      return failure();
    for (size_t index = 0; index < first.size(); ++index)
      if (second[first[index]] != static_cast<int64_t>(index))
        return failure();
    rewriter.replaceOp(op, inner.getInput());
    return success();
  }
};

} // namespace

LogicalResult Conv2DOp::verify() {
  auto input = getF32Tensor(*this, getInput().getType(), "input");
  auto filter = getF32Tensor(*this, getFilter().getType(), "filter");
  auto result = getF32Tensor(*this, getResult().getType(), "result");
  if (failed(input) || failed(filter) || failed(result))
    return failure();
  if ((*input).getRank() != 4 || (*filter).getRank() != 4 ||
      (*result).getRank() != 4)
    return emitOpError("input, filter, and result must have rank 4");
  if (getBias().size() > 1)
    return emitOpError("accepts at most one bias tensor");

  auto strides = getStrides();
  auto pads = getPads();
  auto dilations = getDilations();
  if (strides.size() != 2 || dilations.size() != 2 || pads.size() != 4)
    return emitOpError("expects 2 strides, 4 pads, and 2 dilations");
  if (llvm::any_of(strides, [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(dilations, [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(pads, [](int64_t value) { return value < 0; }))
    return emitOpError(
        "strides/dilations must be positive and pads nonnegative");
  int64_t groups = getGroupsAttr().getInt();
  if (groups <= 0)
    return emitOpError("groups must be positive");
  if (groups != 1)
    return emitOpError("BuddyGraph MVP currently supports groups=1 only");

  bool nchw = getLayout() == Layout::NCHW;
  int64_t inputChannel = (*input).getDimSize(nchw ? 1 : 3);
  int64_t filterOutput = (*filter).getDimSize(0);
  int64_t filterInput = (*filter).getDimSize(nchw ? 1 : 3);
  int64_t resultChannel = (*result).getDimSize(nchw ? 1 : 3);
  if (!areCompatibleDims(filterOutput, resultChannel))
    return emitOpError(
        "result channel count must equal filter output channels");
  if (!ShapedType::isDynamic(inputChannel) && inputChannel % groups != 0)
    return emitOpError("input channels must be divisible by groups");
  if (!ShapedType::isDynamic(filterOutput) && filterOutput % groups != 0)
    return emitOpError("output channels must be divisible by groups");
  if (!ShapedType::isDynamic(inputChannel) &&
      !ShapedType::isDynamic(filterInput) &&
      inputChannel / groups != filterInput)
    return emitOpError("filter input channels do not match input/groups");

  if (!getBias().empty()) {
    auto bias = getF32Tensor(*this, getBias().front().getType(), "bias");
    if (failed(bias))
      return failure();
    if ((*bias).getRank() != 1 ||
        !areCompatibleDims((*bias).getDimSize(0), filterOutput))
      return emitOpError(
          "bias must be rank 1 with one value per output channel");
  }

  int64_t inputH = (*input).getDimSize(nchw ? 2 : 1);
  int64_t inputW = (*input).getDimSize(nchw ? 3 : 2);
  int64_t kernelH = (*filter).getDimSize(nchw ? 2 : 1);
  int64_t kernelW = (*filter).getDimSize(nchw ? 3 : 2);
  auto outputH = inferConvSpatialDim(inputH, kernelH, pads[0], pads[2],
                                     strides[0], dilations[0]);
  auto outputW = inferConvSpatialDim(inputW, kernelW, pads[1], pads[3],
                                     strides[1], dilations[1]);
  if (failed(outputH) || failed(outputW))
    return emitOpError(
        "kernel does not fit within the padded input or dimensions overflow");
  SmallVector<int64_t> expected((*input).getShape().begin(),
                                (*input).getShape().end());
  expected[nchw ? 1 : 3] = filterOutput;
  expected[nchw ? 2 : 1] = *outputH;
  expected[nchw ? 3 : 2] = *outputW;
  if (!areCompatibleShapes(expected, (*result).getShape()))
    return emitOpError() << "result shape " << (*result).getShape()
                         << " does not match inferred shape " << expected;
  return success();
}

LogicalResult BatchNormOp::verify() {
  auto input = getF32Tensor(*this, getInput().getType(), "input");
  auto result = getF32Tensor(*this, getResult().getType(), "result");
  if (failed(input) || failed(result))
    return failure();
  if (!areCompatibleShapes((*input).getShape(), (*result).getShape()))
    return emitOpError("input and result shapes must match");
  if (!(getEpsilon().convertToDouble() > 0.0) ||
      !std::isfinite(getEpsilon().convertToDouble()))
    return emitOpError("epsilon must be finite and positive");
  int64_t channelAxis =
      getLayout() == Layout::NCHW ? 1 : (*input).getRank() - 1;
  if ((*input).getRank() < 2)
    return emitOpError("input must have rank at least 2");
  int64_t channel = (*input).getDimSize(channelAxis);
  SmallVector<std::pair<StringRef, Value>, 4> parameters{
      {"scale", getScale()},
      {"bias", getBias()},
      {"mean", getMean()},
      {"variance", getVariance()}};
  for (auto [name, value] : parameters) {
    auto parameter = getF32Tensor(*this, value.getType(), name);
    if (failed(parameter))
      return failure();
    if ((*parameter).getRank() != 1 ||
        !areCompatibleDims((*parameter).getDimSize(0), channel))
      return emitOpError() << name << " must be rank 1 with the channel length";
  }
  return success();
}

LogicalResult ReluOp::verify() {
  return succeeded(getF32Tensor(*this, getInput().getType(), "input"))
             ? success()
             : failure();
}

LogicalResult ClampOp::verify() {
  auto input = getF32Tensor(*this, getInput().getType(), "input");
  auto result = getF32Tensor(*this, getResult().getType(), "result");
  if (failed(input) || failed(result))
    return failure();
  if (!areCompatibleShapes((*input).getShape(), (*result).getShape()))
    return emitOpError("input and result shapes must match");
  double minimum = getMinValue().convertToDouble();
  double maximum = getMaxValue().convertToDouble();
  if (!std::isfinite(minimum) || !std::isfinite(maximum))
    return emitOpError("bounds must be finite");
  if (minimum > maximum)
    return emitOpError("min_value must be less than or equal to max_value");
  return success();
}

LogicalResult AddOp::verify() { return verifyBinary(*this); }
LogicalResult SubOp::verify() { return verifyBinary(*this); }
LogicalResult MulOp::verify() { return verifyBinary(*this); }
LogicalResult DivOp::verify() { return verifyBinary(*this); }

LogicalResult ReshapeOp::verify() {
  auto input = getF32Tensor(*this, getInput().getType(), "input");
  auto result = getF32Tensor(*this, getResult().getType(), "result");
  if (failed(input) || failed(result))
    return failure();
  auto inferred = inferReshapeShape(*input, getShape());
  if (failed(inferred))
    return emitOpError(
        "shape must contain positive dimensions and at most one valid -1");
  if (!areCompatibleShapes(*inferred, (*result).getShape()))
    return emitOpError() << "result shape " << (*result).getShape()
                         << " does not match requested shape " << *inferred;
  return success();
}

LogicalResult TransposeOp::verify() {
  auto input = getF32Tensor(*this, getInput().getType(), "input");
  auto result = getF32Tensor(*this, getResult().getType(), "result");
  if (failed(input) || failed(result))
    return failure();
  ArrayRef<int64_t> permutation = getPermutation();
  if (permutation.size() != static_cast<size_t>((*input).getRank()))
    return emitOpError("permutation length must equal input rank");
  llvm::SmallBitVector seen((*input).getRank());
  SmallVector<int64_t> expected;
  for (int64_t axis : permutation) {
    if (axis < 0 || axis >= (*input).getRank() || seen[axis])
      return emitOpError(
          "permutation must contain each input axis exactly once");
    seen.set(axis);
    expected.push_back((*input).getDimSize(axis));
  }
  if (!areCompatibleShapes(expected, (*result).getShape()))
    return emitOpError() << "result shape " << (*result).getShape()
                         << " does not match permutation shape " << expected;
  return success();
}

LogicalResult ReduceMeanOp::verify() {
  auto input = getF32Tensor(*this, getInput().getType(), "input");
  auto result = getF32Tensor(*this, getResult().getType(), "result");
  if (failed(input) || failed(result))
    return failure();
  auto expected = inferReduceShape(*input, getAxes(), getKeepDims());
  if (failed(expected))
    return emitOpError("axes must be unique and within the input rank");
  if (!areCompatibleShapes(*expected, (*result).getShape()))
    return emitOpError() << "result shape " << (*result).getShape()
                         << " does not match reduction shape " << *expected;
  return success();
}

LogicalResult FusedElementwiseOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one tensor input");
  auto result = getF32Tensor(*this, getResult().getType(), "result");
  if (failed(result))
    return failure();
  SmallVector<int64_t> broadcastShape;
  for (Value inputValue : getInputs()) {
    auto input = getF32Tensor(*this, inputValue.getType(), "input");
    if (failed(input))
      return failure();
    if (broadcastShape.empty())
      broadcastShape.assign((*input).getShape().begin(),
                            (*input).getShape().end());
    else {
      auto next = inferBroadcastShape(broadcastShape, (*input).getShape());
      if (failed(next))
        return emitOpError("input tensors are not broadcast compatible");
      broadcastShape = std::move(*next);
    }
  }
  if (!areCompatibleShapes(broadcastShape, (*result).getShape()))
    return emitOpError("result shape does not match broadcasted inputs");
  if (getBody().empty() || !getBody().hasOneBlock())
    return emitOpError("body must contain exactly one block");
  Block &block = getBody().front();
  if (block.getNumArguments() != getInputs().size())
    return emitOpError("body must have one scalar argument per tensor input");
  for (BlockArgument argument : block.getArguments())
    if (!argument.getType().isF32())
      return emitOpError("body arguments must be f32 scalars");
  for (Operation &nested : block) {
    if (isa<YieldOp, arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
            arith::MaximumFOp, arith::CmpFOp, arith::SelectOp,
            arith::ConstantOp>(nested))
      continue;
    return emitOpError() << "body contains unsupported scalar operation "
                         << nested.getName();
  }
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield || !yield.getValue().getType().isF32())
    return emitOpError("body must terminate with bgraph.yield of f32");
  return success();
}

LogicalResult YieldOp::verify() {
  auto parent = (*this)->getParentOfType<FusedElementwiseOp>();
  if (!parent)
    return emitOpError("must be nested in bgraph.fused_elementwise");
  auto result = dyn_cast<RankedTensorType>(parent.getResult().getType());
  if (!result || getValue().getType() != result.getElementType())
    return emitOpError("value type must equal the parent result element type");
  return success();
}

void ReluOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add<ElideNestedRelu>(context);
}

void ClampOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *context) {
  patterns.add<ElideNestedClamp>(context);
}

void AddOp::getCanonicalizationPatterns(RewritePatternSet &, MLIRContext *) {}
void SubOp::getCanonicalizationPatterns(RewritePatternSet &, MLIRContext *) {}
void MulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<ElideMulByOne>(context);
}
void DivOp::getCanonicalizationPatterns(RewritePatternSet &, MLIRContext *) {}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add<CollapseReshape>(context);
}

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<CancelTranspose>(context);
}

#define GET_OP_CLASSES
#include "BuddyGraph/IR/BGraphOps.cpp.inc"
