#include "BuddyGraph/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <functional>

using namespace mlir;

namespace buddy::bgraph {
#define GEN_PASS_DEF_BGRAPHVECTORIZECONV
#include "BuddyGraph/Transforms/Passes.h.inc"

namespace {

Value index(OpBuilder &b, Location loc, int64_t value) {
  return b.create<arith::ConstantIndexOp>(loc, value);
}

void loop(OpBuilder &b, Location loc, int64_t begin, int64_t end, int64_t step,
          function_ref<void(OpBuilder &, Value)> body) {
  if (begin == end)
    return;
  b.create<scf::ForOp>(
      loc, index(b, loc, begin), index(b, loc, end), index(b, loc, step),
      ValueRange{}, [&](OpBuilder &nested, Location, Value iv, ValueRange) {
        body(nested, iv);
        nested.create<scf::YieldOp>(loc);
      });
}

bool hasUnitInnerStride(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(type.getStridesAndOffset(strides, offset)) &&
         strides.back() == 1;
}

// Buffer-level scheduling: keep each output tile in vector SSA accumulators.
// Each input vector is reused across channelTile independent output channels.
class VectorizeConvPattern final
    : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
public:
  VectorizeConvPattern(MLIRContext *context, int64_t width, int64_t channels)
      : OpRewritePattern(context), width(width), channels(channels) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics())
      return failure();
    Value input = op.getInputs()[0], filter = op.getInputs()[1];
    Value output = op.getOutputs()[0];
    auto inputType = cast<MemRefType>(input.getType());
    auto filterType = cast<MemRefType>(filter.getType());
    auto outputType = cast<MemRefType>(output.getType());
    if (!inputType.hasStaticShape() || !filterType.hasStaticShape() ||
        !outputType.hasStaticShape() || !inputType.getElementType().isF32() ||
        !filterType.getElementType().isF32() ||
        !outputType.getElementType().isF32())
      return failure();
    SmallVector<int64_t> strides(op.getStrides().getValues<int64_t>());
    SmallVector<int64_t> dilations(op.getDilations().getValues<int64_t>());
    // Adjacent output pixels must correspond to adjacent input pixels.
    // Other layouts/strides stay on the standard Linalg lowering path.
    if (strides[1] != 1 || !hasUnitInnerStride(inputType) ||
        !hasUnitInnerStride(outputType))
      return failure();

    Location loc = op.getLoc();
    int64_t batches = outputType.getDimSize(0);
    int64_t outChannels = outputType.getDimSize(1);
    int64_t outHeight = outputType.getDimSize(2);
    int64_t outWidth = outputType.getDimSize(3);
    int64_t fullWidth = outWidth / width * width;
    int64_t fullChannels = outChannels / channels * channels;
    auto vectorType = VectorType::get({width}, rewriter.getF32Type());

    auto add = [&](OpBuilder &b, Value lhs, Value rhs) -> Value {
      return b.create<arith::AddIOp>(loc, lhs, rhs);
    };
    auto scaled = [&](OpBuilder &b, Value value, int64_t scale) -> Value {
      return b.create<arith::MulIOp>(loc, value, index(b, loc, scale));
    };

    auto emitTile = [&](OpBuilder &b, Value n, Value oc, Value oh, Value ow,
                        int64_t count, bool vectorized) {
      SmallVector<Value> outputChannels, initial;
      for (int64_t i = 0; i < count; ++i) {
        Value channel = add(b, oc, index(b, loc, i));
        outputChannels.push_back(channel);
        SmallVector<Value> at{n, channel, oh, ow};
        initial.push_back(vectorized
                              ? Value(b.create<vector::LoadOp>(loc, vectorType,
                                                               output, at))
                              : Value(b.create<memref::LoadOp>(loc, output, at)));
      }

      // Preserve the original ic -> kh -> kw reduction order, with separate
      // multiply/add operations (no reassociation or implicit FMA contraction).
      std::function<SmallVector<Value>(OpBuilder &, unsigned, ValueRange,
                                       SmallVector<Value>)>
          reduce;
      reduce = [&](OpBuilder &builder, unsigned dim, ValueRange accumulators,
                   SmallVector<Value> ivs) -> SmallVector<Value> {
        if (dim != 3) {
          auto reduction = builder.create<scf::ForOp>(
              loc, index(builder, loc, 0),
              index(builder, loc, filterType.getDimSize(dim + 1)),
              index(builder, loc, 1), accumulators,
              [&](OpBuilder &nested, Location, Value iv, ValueRange carried) {
                auto nextIVs = ivs;
                nextIVs.push_back(iv);
                nested.create<scf::YieldOp>(
                    loc, reduce(nested, dim + 1, carried, nextIVs));
              });
          return SmallVector<Value>(reduction.getResults());
        }
        Value ih = add(builder, scaled(builder, oh, strides[0]),
                       scaled(builder, ivs[1], dilations[0]));
        Value iw = add(builder, ow, scaled(builder, ivs[2], dilations[1]));
        SmallVector<Value> at{n, ivs[0], ih, iw};
        Value x = vectorized
                      ? Value(builder.create<vector::LoadOp>(loc, vectorType,
                                                             input, at))
                      : Value(builder.create<memref::LoadOp>(loc, input, at));
        SmallVector<Value> next;
        for (int64_t i = 0; i < count; ++i) {
          Value weight = builder.create<memref::LoadOp>(
              loc, filter,
              ValueRange{outputChannels[i], ivs[0], ivs[1], ivs[2]});
          if (vectorized)
            weight = builder.create<vector::BroadcastOp>(loc, vectorType, weight);
          Value product = builder.create<arith::MulFOp>(loc, x, weight);
          next.push_back(builder.create<arith::AddFOp>(loc, accumulators[i],
                                                       product));
        }
        return next;
      };
      auto result = reduce(b, 0, initial, {});
      for (int64_t i = 0; i < count; ++i) {
        SmallVector<Value> at{n, outputChannels[i], oh, ow};
        if (vectorized)
          b.create<vector::StoreOp>(loc, result[i], output, at);
        else
          b.create<memref::StoreOp>(loc, result[i], output, at);
      }
    };

    loop(rewriter, loc, 0, batches, 1, [&](OpBuilder &b, Value n) {
      auto emitRows = [&](OpBuilder &builder, Value oc, int64_t count) {
        loop(builder, loc, 0, outHeight, 1, [&](OpBuilder &row, Value oh) {
          loop(row, loc, 0, fullWidth, width, [&](OpBuilder &tile, Value ow) {
            emitTile(tile, n, oc, oh, ow, count, true);
          });
          loop(row, loc, fullWidth, outWidth, 1, [&](OpBuilder &tail, Value ow) {
            emitTile(tail, n, oc, oh, ow, count, false);
          });
        });
      };
      loop(b, loc, 0, fullChannels, channels, [&](OpBuilder &tile, Value oc) {
        emitRows(tile, oc, channels);
      });
      if (outChannels != fullChannels)
        emitRows(b, index(b, loc, fullChannels), outChannels - fullChannels);
    });
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t width, channels;
};

class VectorizeConvPass
    : public impl::BGraphVectorizeConvBase<VectorizeConvPass> {
public:
  using Base::Base;
  void runOnOperation() final {
    if (vectorWidth < 1 || channelTile < 1) {
      getOperation().emitError("vector-width and channel-tile must be positive");
      return signalPassFailure();
    }
    RewritePatternSet patterns(&getContext());
    patterns.add<VectorizeConvPattern>(&getContext(), vectorWidth, channelTile);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createVectorizeConvPass() {
  return std::make_unique<VectorizeConvPass>();
}

void registerCPUPipelines() {
  PassPipelineRegistration<>(
      "bgraph-bufferize",
      "Propagate caller output destinations, bufferize and release temporaries",
      [](OpPassManager &pm) {
        pm.addPass(createInlinerPass());
        pm.addPass(createSymbolDCEPass());
        pm.addPass(bufferization::createEmptyTensorEliminationPass());
        bufferization::OneShotBufferizePassOptions options;
        options.bufferizeFunctionBoundaries = true;
        pm.addPass(bufferization::createOneShotBufferizePass(options));
        bufferization::buildBufferDeallocationPipeline(
            pm, bufferization::BufferDeallocationPipelineOptions{});
      });
}

} // namespace buddy::bgraph
