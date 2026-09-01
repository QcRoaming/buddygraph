#include "BuddyGraph/Conversion/BGraphToLinalg.h"

#include "BuddyGraph/IR/BGraphDialect.h"
#include "BuddyGraph/IR/BGraphOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;

namespace buddy::bgraph {
#define GEN_PASS_DEF_CONVERTBGRAPHTOLINALG
#include "BuddyGraph/Transforms/Passes.h.inc"

namespace {

SmallVector<utils::IteratorType> parallelIterators(int64_t rank) {
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

AffineMap broadcastMap(RankedTensorType inputType, RankedTensorType resultType,
                       Builder &builder) {
  int64_t resultRank = resultType.getRank();
  int64_t inputRank = inputType.getRank();
  SmallVector<AffineExpr> expressions;
  expressions.reserve(inputRank);
  for (int64_t inputDim = 0; inputDim < inputRank; ++inputDim) {
    int64_t resultDim = resultRank - inputRank + inputDim;
    bool broadcasts = inputType.getDimSize(inputDim) == 1 &&
                      resultType.getDimSize(resultDim) != 1;
    expressions.push_back(broadcasts ? builder.getAffineConstantExpr(0)
                                     : builder.getAffineDimExpr(resultDim));
  }
  return AffineMap::get(resultRank, 0, expressions, builder.getContext());
}

template <typename BodyBuilder>
FailureOr<Value>
createElementwiseGeneric(Location location, RankedTensorType resultType,
                         ValueRange inputs, ConversionPatternRewriter &rewriter,
                         BodyBuilder &&bodyBuilder) {
  if (!resultType.hasStaticShape() || llvm::any_of(inputs, [](Value input) {
        return !cast<RankedTensorType>(input.getType()).hasStaticShape();
      }))
    return failure();

  Value output = rewriter.create<tensor::EmptyOp>(
      location, resultType.getShape(), resultType.getElementType());
  SmallVector<AffineMap> maps;
  maps.reserve(inputs.size() + 1);
  for (Value input : inputs)
    maps.push_back(broadcastMap(cast<RankedTensorType>(input.getType()),
                                resultType, rewriter));
  maps.push_back(rewriter.getMultiDimIdentityMap(resultType.getRank()));

  auto generic = rewriter.create<linalg::GenericOp>(
      location, resultType, inputs, output, maps,
      parallelIterators(resultType.getRank()),
      [&](OpBuilder &builder, Location nestedLocation, ValueRange arguments) {
        Value value = bodyBuilder(builder, nestedLocation,
                                  arguments.take_front(inputs.size()));
        builder.create<linalg::YieldOp>(nestedLocation, value);
      });
  return generic.getResult(0);
}

template <typename OpTy, typename ScalarOp>
class BinaryLowering final : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto result = createElementwiseGeneric(
        op.getLoc(), resultType, adaptor.getOperands(), rewriter,
        [](OpBuilder &builder, Location location, ValueRange arguments) {
          return builder.create<ScalarOp>(location, arguments[0], arguments[1])
              .getResult();
        });
    if (failed(result))
      return rewriter.notifyMatchFailure(op,
                                         "requires statically shaped tensors");
    rewriter.replaceOp(op, *result);
    return success();
  }
};

class ReluLowering final : public OpConversionPattern<ReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto result = createElementwiseGeneric(
        op.getLoc(), resultType, adaptor.getOperands(), rewriter,
        [&](OpBuilder &builder, Location location, ValueRange arguments) {
          Value zero = builder.create<arith::ConstantFloatOp>(
              location, APFloat(0.0f), builder.getF32Type());
          return builder.create<arith::MaximumFOp>(location, arguments[0], zero)
              .getResult();
        });
    if (failed(result))
      return rewriter.notifyMatchFailure(op,
                                         "requires statically shaped tensors");
    rewriter.replaceOp(op, *result);
    return success();
  }
};

class ClampLowering final : public OpConversionPattern<ClampOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ClampOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto result = createElementwiseGeneric(
        op.getLoc(), resultType, adaptor.getOperands(), rewriter,
        [&](OpBuilder &builder, Location location, ValueRange arguments) {
          Value minimum = builder.create<arith::ConstantFloatOp>(
              location, op.getMinValue(), builder.getF32Type());
          Value maximum = builder.create<arith::ConstantFloatOp>(
              location, op.getMaxValue(), builder.getF32Type());
          Value below = builder.create<arith::CmpFOp>(
              location, arith::CmpFPredicate::OLT, arguments[0], minimum);
          Value lowerClamped = builder.create<arith::SelectOp>(
              location, below, minimum, arguments[0]);
          Value above = builder.create<arith::CmpFOp>(
              location, arith::CmpFPredicate::OLT, maximum, lowerClamped);
          return builder
              .create<arith::SelectOp>(location, above, maximum, lowerClamped)
              .getResult();
        });
    if (failed(result))
      return rewriter.notifyMatchFailure(op,
                                         "requires statically shaped tensors");
    rewriter.replaceOp(op, *result);
    return success();
  }
};

class FusedElementwiseLowering final
    : public OpConversionPattern<FusedElementwiseOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FusedElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Block &sourceBlock = op.getBody().front();
    auto result = createElementwiseGeneric(
        op.getLoc(), resultType, adaptor.getOperands(), rewriter,
        [&](OpBuilder &builder, Location location, ValueRange arguments) {
          IRMapping mapping;
          for (auto [source, target] :
               llvm::zip_equal(sourceBlock.getArguments(), arguments))
            mapping.map(source, target);
          for (Operation &nested : sourceBlock.without_terminator())
            builder.clone(nested, mapping);
          auto sourceYield = cast<YieldOp>(sourceBlock.getTerminator());
          return mapping.lookup(sourceYield.getValue());
        });
    if (failed(result))
      return rewriter.notifyMatchFailure(op,
                                         "requires statically shaped tensors");
    rewriter.replaceOp(op, *result);
    return success();
  }
};

class ReshapeLowering final : public OpConversionPattern<ReshapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    if (!resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "requires a static result shape");
    auto shapeType =
        RankedTensorType::get({resultType.getRank()}, rewriter.getI64Type());
    SmallVector<int64_t> shape(resultType.getShape());
    auto shapeAttr = DenseIntElementsAttr::get(shapeType, shape);
    Value shapeValue =
        rewriter.create<arith::ConstantOp>(op.getLoc(), shapeType, shapeAttr);
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(
        op, resultType, adaptor.getInput(), shapeValue);
    return success();
  }
};

class TransposeLowering final : public OpConversionPattern<TransposeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    if (!resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "requires a static result shape");
    Value output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        op, adaptor.getInput(), output, op.getPermutation());
    return success();
  }
};

class ReduceMeanLowering final : public OpConversionPattern<ReduceMeanOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceMeanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    if (!inputType.hasStaticShape() || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "requires static shapes");

    llvm::SmallBitVector reduced(inputType.getRank());
    int64_t elementCount = 1;
    for (int64_t axis : op.getAxes()) {
      int64_t normalized = axis < 0 ? axis + inputType.getRank() : axis;
      reduced.set(normalized);
      elementCount *= inputType.getDimSize(normalized);
    }

    Value empty = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());
    Value zero = rewriter.create<arith::ConstantFloatOp>(
        op.getLoc(), APFloat(0.0f), rewriter.getF32Type());
    Value initialized =
        rewriter.create<linalg::FillOp>(op.getLoc(), zero, empty).result();

    SmallVector<AffineExpr> outputExpressions;
    for (int64_t index = 0; index < inputType.getRank(); ++index) {
      if (!reduced[index])
        outputExpressions.push_back(rewriter.getAffineDimExpr(index));
      else if (op.getKeepDims())
        outputExpressions.push_back(rewriter.getAffineConstantExpr(0));
    }
    SmallVector<AffineMap> reductionMaps{
        rewriter.getMultiDimIdentityMap(inputType.getRank()),
        AffineMap::get(inputType.getRank(), 0, outputExpressions,
                       rewriter.getContext())};
    SmallVector<utils::IteratorType> iterators;
    for (int64_t index = 0; index < inputType.getRank(); ++index)
      iterators.push_back(reduced[index] ? utils::IteratorType::reduction
                                         : utils::IteratorType::parallel);
    Value sum = rewriter
                    .create<linalg::GenericOp>(
                        op.getLoc(), resultType, adaptor.getInput(),
                        initialized, reductionMaps, iterators,
                        [](OpBuilder &builder, Location location,
                           ValueRange arguments) {
                          Value added = builder.create<arith::AddFOp>(
                              location, arguments[0], arguments[1]);
                          builder.create<linalg::YieldOp>(location, added);
                        })
                    .getResult(0);

    auto mean = createElementwiseGeneric(
        op.getLoc(), resultType, sum, rewriter,
        [elementCount](OpBuilder &builder, Location location,
                       ValueRange arguments) {
          Value divisor = builder.create<arith::ConstantFloatOp>(
              location, APFloat(static_cast<float>(elementCount)),
              builder.getF32Type());
          return builder.create<arith::DivFOp>(location, arguments[0], divisor)
              .getResult();
        });
    if (failed(mean))
      return failure();
    rewriter.replaceOp(op, *mean);
    return success();
  }
};

class BatchNormLowering final : public OpConversionPattern<BatchNormOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(BatchNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    if (!resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "requires a static result shape");
    int64_t channelAxis =
        op.getLayout() == Layout::NCHW ? 1 : resultType.getRank() - 1;
    AffineMap identity = rewriter.getMultiDimIdentityMap(resultType.getRank());
    AffineMap channelMap = AffineMap::get(
        resultType.getRank(), 0, {rewriter.getAffineDimExpr(channelAxis)},
        rewriter.getContext());
    SmallVector<AffineMap> maps{identity,   channelMap, channelMap,
                                channelMap, channelMap, identity};
    Value output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());
    SmallVector<Value> inputs(adaptor.getOperands().begin(),
                              adaptor.getOperands().end());
    auto generic = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, inputs, output, maps,
        parallelIterators(resultType.getRank()),
        [&](OpBuilder &builder, Location location, ValueRange arguments) {
          Value centered = builder.create<arith::SubFOp>(location, arguments[0],
                                                         arguments[3]);
          Value epsilon = builder.create<arith::ConstantFloatOp>(
              location,
              APFloat(static_cast<float>(op.getEpsilon().convertToDouble())),
              builder.getF32Type());
          Value variance =
              builder.create<arith::AddFOp>(location, arguments[4], epsilon);
          Value denominator = builder.create<math::SqrtOp>(location, variance);
          Value normalized =
              builder.create<arith::DivFOp>(location, centered, denominator);
          Value scaled =
              builder.create<arith::MulFOp>(location, normalized, arguments[1]);
          Value shifted =
              builder.create<arith::AddFOp>(location, scaled, arguments[2]);
          builder.create<linalg::YieldOp>(location, shifted);
        });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

Value applyPadding(Location location, Value input, ArrayRef<int64_t> pads,
                   bool nchw, ConversionPatternRewriter &rewriter) {
  if (llvm::all_of(pads, [](int64_t pad) { return pad == 0; }))
    return input;
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<int64_t> paddedShape(inputType.getShape());
  int64_t heightAxis = nchw ? 2 : 1;
  int64_t widthAxis = nchw ? 3 : 2;
  paddedShape[heightAxis] += pads[0] + pads[2];
  paddedShape[widthAxis] += pads[1] + pads[3];
  SmallVector<OpFoldResult> low(4, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> high(4, rewriter.getIndexAttr(0));
  low[heightAxis] = rewriter.getIndexAttr(pads[0]);
  low[widthAxis] = rewriter.getIndexAttr(pads[1]);
  high[heightAxis] = rewriter.getIndexAttr(pads[2]);
  high[widthAxis] = rewriter.getIndexAttr(pads[3]);
  Value zero = rewriter.create<arith::ConstantFloatOp>(location, APFloat(0.0f),
                                                       rewriter.getF32Type());
  return rewriter.create<tensor::PadOp>(
      location, RankedTensorType::get(paddedShape, inputType.getElementType()),
      input, low, high, zero);
}

class Conv2DLowering final : public OpConversionPattern<Conv2DOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Conv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto hasStaticTensorType = [](Value value) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      return type && type.hasStaticShape();
    };
    if (!resultType.hasStaticShape() || !hasStaticTensorType(op.getInput()) ||
        !hasStaticTensorType(op.getFilter()) ||
        llvm::any_of(op.getBias(),
                     [&](Value bias) { return !hasStaticTensorType(bias); }))
      return rewriter.notifyMatchFailure(
          op, "requires static input, filter, bias, and result shapes");
    bool nchw = op.getLayout() == Layout::NCHW;
    Value input = applyPadding(op.getLoc(), adaptor.getInput(), op.getPads(),
                               nchw, rewriter);
    Value empty = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());
    Value initialized;
    if (adaptor.getBias().empty()) {
      Value zero = rewriter.create<arith::ConstantFloatOp>(
          op.getLoc(), APFloat(0.0f), rewriter.getF32Type());
      initialized =
          rewriter.create<linalg::FillOp>(op.getLoc(), zero, empty).result();
    } else {
      int64_t channelAxis = nchw ? 1 : 3;
      AffineMap channelMap =
          AffineMap::get(4, 0, {rewriter.getAffineDimExpr(channelAxis)},
                         rewriter.getContext());
      SmallVector<AffineMap> maps{channelMap,
                                  rewriter.getMultiDimIdentityMap(4)};
      initialized =
          rewriter
              .create<linalg::GenericOp>(
                  op.getLoc(), resultType, adaptor.getBias().front(), empty,
                  maps, parallelIterators(4),
                  [](OpBuilder &builder, Location location,
                     ValueRange arguments) {
                    builder.create<linalg::YieldOp>(location, arguments[0]);
                  })
              .getResult(0);
    }
    auto strides = rewriter.getI64TensorAttr(op.getStrides());
    auto dilations = rewriter.getI64TensorAttr(op.getDilations());
    Value result;
    if (nchw)
      result = rewriter
                   .create<linalg::Conv2DNchwFchwOp>(
                       op.getLoc(), resultType,
                       ValueRange{input, adaptor.getFilter()}, initialized,
                       strides, dilations)
                   .getResult(0);
    else
      result = rewriter
                   .create<linalg::Conv2DNhwcFhwcOp>(
                       op.getLoc(), resultType,
                       ValueRange{input, adaptor.getFilter()}, initialized,
                       strides, dilations)
                   .getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ConvertBGraphToLinalg
    : public impl::ConvertBGraphToLinalgBase<ConvertBGraphToLinalg> {
public:
  using Base::Base;
  void runOnOperation() final {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           linalg::LinalgDialect, math::MathDialect,
                           tensor::TensorDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<BGraphDialect>();

    RewritePatternSet patterns(&getContext());
    patterns
        .add<BinaryLowering<AddOp, arith::AddFOp>,
             BinaryLowering<SubOp, arith::SubFOp>,
             BinaryLowering<MulOp, arith::MulFOp>,
             BinaryLowering<DivOp, arith::DivFOp>, ReluLowering, ClampLowering,
             FusedElementwiseLowering, ReshapeLowering, TransposeLowering,
             ReduceMeanLowering, BatchNormLowering, Conv2DLowering>(
            &getContext());
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertBGraphToLinalgPass() {
  return std::make_unique<ConvertBGraphToLinalg>();
}

} // namespace buddy::bgraph
