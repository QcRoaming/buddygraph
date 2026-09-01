#include "BuddyGraph/Transforms/Passes.h"

#include "BuddyGraph/IR/BGraphDialect.h"
#include "BuddyGraph/IR/BGraphOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cmath>

using namespace mlir;

namespace buddy::bgraph {
#define GEN_PASS_DEF_BGRAPHFOLDBATCHNORMINTOCONV
#include "BuddyGraph/Transforms/Passes.h.inc"

namespace {

FailureOr<DenseFPElementsAttr> getDenseF32(Value value) {
  Attribute attribute;
  if (!matchPattern(value, m_Constant(&attribute)))
    return failure();
  auto dense = dyn_cast<DenseFPElementsAttr>(attribute);
  if (!dense || !dense.getElementType().isF32())
    return failure();
  return dense;
}

SmallVector<float> asFloats(DenseFPElementsAttr values) {
  SmallVector<float> result;
  result.reserve(values.getNumElements());
  for (const APFloat &value : values.getValues<APFloat>())
    result.push_back(value.convertToFloat());
  return result;
}

class FoldBatchNormPattern final : public OpRewritePattern<BatchNormOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchNormOp batchNorm,
                                PatternRewriter &rewriter) const override {
    auto conv = batchNorm.getInput().getDefiningOp<Conv2DOp>();
    if (!conv)
      return rewriter.notifyMatchFailure(batchNorm, "input is not Conv2D");
    if (!conv.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(batchNorm,
                                         "Conv2D result has multiple users");
    if (conv.getGroupsAttr().getInt() != 1)
      return rewriter.notifyMatchFailure(batchNorm,
                                         "MVP folding requires groups=1");
    if (conv.getLayout() != batchNorm.getLayout())
      return rewriter.notifyMatchFailure(
          batchNorm, "Conv2D and BatchNorm layouts must match");

    auto filterAttr = getDenseF32(conv.getFilter());
    auto scaleAttr = getDenseF32(batchNorm.getScale());
    auto betaAttr = getDenseF32(batchNorm.getBias());
    auto meanAttr = getDenseF32(batchNorm.getMean());
    auto varianceAttr = getDenseF32(batchNorm.getVariance());
    if (failed(filterAttr) || failed(scaleAttr) || failed(betaAttr) ||
        failed(meanAttr) || failed(varianceAttr))
      return rewriter.notifyMatchFailure(
          batchNorm,
          "filter and BatchNorm parameters must be dense f32 constants");

    auto filterType = cast<RankedTensorType>(conv.getFilter().getType());
    int64_t channels = filterType.getDimSize(0);
    if (channels <= 0 || (*scaleAttr).getNumElements() != channels ||
        (*betaAttr).getNumElements() != channels ||
        (*meanAttr).getNumElements() != channels ||
        (*varianceAttr).getNumElements() != channels)
      return rewriter.notifyMatchFailure(batchNorm,
                                         "parameter element counts disagree");

    SmallVector<float> oldBias(channels, 0.0f);
    if (!conv.getBias().empty()) {
      auto biasAttr = getDenseF32(conv.getBias().front());
      if (failed(biasAttr) || (*biasAttr).getNumElements() != channels)
        return rewriter.notifyMatchFailure(
            batchNorm, "Conv2D bias must be a dense f32 channel vector");
      oldBias = asFloats(*biasAttr);
    }

    SmallVector<float> filter = asFloats(*filterAttr);
    SmallVector<float> scale = asFloats(*scaleAttr);
    SmallVector<float> beta = asFloats(*betaAttr);
    SmallVector<float> mean = asFloats(*meanAttr);
    SmallVector<float> variance = asFloats(*varianceAttr);
    SmallVector<float> alpha(channels);
    SmallVector<float> newBias(channels);
    float epsilon =
        static_cast<float>(batchNorm.getEpsilon().convertToDouble());
    for (int64_t channel = 0; channel < channels; ++channel) {
      float denominator = variance[channel] + epsilon;
      if (!(denominator > 0.0) || !std::isfinite(denominator))
        return rewriter.notifyMatchFailure(
            batchNorm, "variance + epsilon is not positive");
      alpha[channel] = scale[channel] / std::sqrt(denominator);
      newBias[channel] =
          beta[channel] + (oldBias[channel] - mean[channel]) * alpha[channel];
    }

    int64_t valuesPerChannel = filter.size() / channels;
    for (int64_t channel = 0; channel < channels; ++channel)
      for (int64_t index = 0; index < valuesPerChannel; ++index)
        filter[channel * valuesPerChannel + index] *= alpha[channel];

    auto newFilterAttr =
        DenseElementsAttr::get(filterType, ArrayRef<float>(filter));
    auto biasType = RankedTensorType::get({channels}, rewriter.getF32Type());
    auto newBiasAttr =
        DenseElementsAttr::get(biasType, ArrayRef<float>(newBias));
    Location location =
        rewriter.getFusedLoc({conv.getLoc(), batchNorm.getLoc()});
    auto newFilter =
        rewriter.create<arith::ConstantOp>(location, filterType, newFilterAttr);
    auto newBiasConstant =
        rewriter.create<arith::ConstantOp>(location, biasType, newBiasAttr);

    OperationState state(location, Conv2DOp::getOperationName());
    SmallVector<Value> convOperands{conv.getInput(), newFilter.getResult(),
                                    newBiasConstant.getResult()};
    state.addOperands(convOperands);
    state.addAttributes(conv->getAttrs());
    state.addTypes(batchNorm.getResult().getType());
    auto newConv = cast<Conv2DOp>(rewriter.create(state));
    rewriter.replaceOp(batchNorm, newConv.getResult());
    rewriter.eraseOp(conv);
    return success();
  }
};

class BGraphFoldBatchNormIntoConv
    : public impl::BGraphFoldBatchNormIntoConvBase<
          BGraphFoldBatchNormIntoConv> {
public:
  using Base::Base;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldBatchNormPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFoldBatchNormIntoConvPass() {
  return std::make_unique<BGraphFoldBatchNormIntoConv>();
}

} // namespace buddy::bgraph
