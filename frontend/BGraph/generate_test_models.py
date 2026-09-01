#!/usr/bin/env python3
"""Generate deterministic ONNX fixtures for BuddyGraph importer and E2E tests."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _save(graph: onnx.GraphProto, path: pathlib.Path, opset: int = 18) -> None:
    model = helper.make_model(
        graph,
        producer_name="buddygraph-tests",
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _tensor(name: str, values: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values), name=name)


def generate(output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 5, 5])
    conv_initializers = [
        _tensor("w", np.arange(4 * 3 * 3 * 3, dtype=np.float32).reshape(4, 3, 3, 3) / 100.0),
        _tensor("conv_bias", np.zeros(4, dtype=np.float32)),
        _tensor("scale", np.ones(4, dtype=np.float32)),
        _tensor("bn_bias", np.zeros(4, dtype=np.float32)),
        _tensor("mean", np.zeros(4, dtype=np.float32)),
        _tensor("variance", np.ones(4, dtype=np.float32)),
    ]
    conv_nodes = [
        helper.make_node(
            "Conv", ["x", "w", "conv_bias"], ["conv"], name="conv", pads=[1, 1, 1, 1]
        ),
        helper.make_node(
            "BatchNormalization",
            ["conv", "scale", "bn_bias", "mean", "variance"],
            ["normalized"],
            name="batch_norm",
            epsilon=1.0e-5,
        ),
        helper.make_node("Relu", ["normalized"], ["y"], name="relu"),
    ]
    _save(
        helper.make_graph(conv_nodes, "conv_bn_relu", [x], [y], initializer=conv_initializers),
        output_dir / "conv_bn_relu.onnx",
    )

    lhs = helper.make_tensor_value_info("lhs", TensorProto.FLOAT, [2, 3])
    rhs = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, [3])
    elementwise_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    elementwise_nodes = [
        helper.make_node("Add", ["lhs", "rhs"], ["sum"], name="broadcast_add"),
        helper.make_node("Sub", ["sum", "rhs"], ["difference"], name="broadcast_sub"),
        helper.make_node("Div", ["difference", "rhs"], ["quotient"], name="broadcast_div"),
        helper.make_node("Relu", ["quotient"], ["activated"], name="relu"),
        helper.make_node("Reshape", ["activated", "reshape_shape"], ["reshaped"], name="reshape"),
        helper.make_node("Transpose", ["reshaped"], ["y"], name="transpose", perm=[1, 0]),
    ]
    _save(
        helper.make_graph(
            elementwise_nodes,
            "elementwise_chain",
            [lhs, rhs],
            [elementwise_y],
            initializer=[_tensor("reshape_shape", np.array([3, 2], dtype=np.int64))],
        ),
        output_dir / "elementwise_chain.onnx",
    )

    constant_y = helper.make_tensor_value_info("result", TensorProto.FLOAT, [])
    epsilon = 1.0e-5
    constant_initializers = [
        _tensor(
            "a",
            np.array([[[[-2.0, 1.0], [3.0, 5.0]]]], dtype=np.float32),
        ),
        _tensor("w", np.ones((1, 1, 1, 1), dtype=np.float32)),
        _tensor("conv_bias", np.zeros(1, dtype=np.float32)),
        _tensor("bn_scale", np.array([np.sqrt(1.0 + epsilon)], dtype=np.float32)),
        _tensor("bn_bias", np.zeros(1, dtype=np.float32)),
        _tensor("bn_mean", np.zeros(1, dtype=np.float32)),
        _tensor("bn_variance", np.ones(1, dtype=np.float32)),
        _tensor("b", np.array([1.0, 2.0], dtype=np.float32)),
        _tensor("factor", np.array(2.0, dtype=np.float32)),
        _tensor("axes", np.array([0, 1, 2, 3], dtype=np.int64)),
    ]
    constant_nodes = [
        helper.make_node("Conv", ["a", "w", "conv_bias"], ["conv"], name="conv"),
        helper.make_node(
            "BatchNormalization",
            ["conv", "bn_scale", "bn_bias", "bn_mean", "bn_variance"],
            ["normalized"],
            name="batch_norm",
            epsilon=epsilon,
        ),
        helper.make_node("Add", ["normalized", "b"], ["sum"], name="add"),
        helper.make_node("Relu", ["sum"], ["activated"], name="relu"),
        helper.make_node("Mul", ["activated", "factor"], ["scaled"], name="mul"),
        helper.make_node(
            "ReduceMean", ["scaled", "axes"], ["result"], name="mean", keepdims=0
        ),
    ]
    _save(
        helper.make_graph(
            constant_nodes,
            "elementwise_constant",
            [],
            [constant_y],
            initializer=constant_initializers,
        ),
        output_dir / "elementwise_constant.onnx",
    )

    clamp_y = helper.make_tensor_value_info("result", TensorProto.FLOAT, [])
    clamp_initializers = [
        _tensor("a", np.array([-3.0, -0.5, 0.5, 4.0], dtype=np.float32)),
        _tensor("b", np.array(0.5, dtype=np.float32)),
        _tensor("clip_min", np.array(-1.0, dtype=np.float32)),
        _tensor("clip_max", np.array(1.0, dtype=np.float32)),
        _tensor("factor", np.array(2.0, dtype=np.float32)),
        _tensor("axes", np.array([0], dtype=np.int64)),
    ]
    clamp_nodes = [
        helper.make_node("Add", ["a", "b"], ["sum"], name="add"),
        helper.make_node(
            "Clip", ["sum", "clip_min", "clip_max"], ["clamped"], name="clip"
        ),
        helper.make_node("Mul", ["clamped", "factor"], ["scaled"], name="mul"),
        helper.make_node(
            "ReduceMean", ["scaled", "axes"], ["result"], name="mean", keepdims=0
        ),
    ]
    _save(
        helper.make_graph(
            clamp_nodes,
            "clamp_constant",
            [],
            [clamp_y],
            initializer=clamp_initializers,
        ),
        output_dir / "clamp_constant.onnx",
    )

    _save(
        helper.make_graph(
            [
                helper.make_node(
                    "Add", ["nan_input", "nan_bias"], ["nan_sum"], name="nan_add"
                ),
                helper.make_node(
                    "Clip", ["nan_sum", "clip_min", "clip_max"], ["clamped"], name="clip"
                ),
                helper.make_node(
                    "ReduceMean", ["clamped", "axes"], ["result"], name="mean", keepdims=0
                ),
            ],
            "clamp_nan",
            [],
            [clamp_y],
            initializer=[
                _tensor("nan_input", np.array([np.nan], dtype=np.float32)),
                _tensor("nan_bias", np.array(0.0, dtype=np.float32)),
                _tensor("clip_min", np.array(-1.0, dtype=np.float32)),
                _tensor("clip_max", np.array(1.0, dtype=np.float32)),
                _tensor("axes", np.array([0], dtype=np.int64)),
            ],
        ),
        output_dir / "clamp_nan.onnx",
    )

    shared_bound_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
    _save(
        helper.make_graph(
            [
                helper.make_node("Add", ["x", "clip_max"], ["sum"], name="add_bound"),
                helper.make_node(
                    "Clip", ["sum", "clip_min", "clip_max"], ["y"], name="clip"
                ),
            ],
            "clamp_shared_initializer",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
            [shared_bound_y],
            initializer=[
                _tensor("clip_min", np.array(-1.0, dtype=np.float32)),
                _tensor("clip_max", np.array(1.0, dtype=np.float32)),
            ],
        ),
        output_dir / "clamp_shared_initializer.onnx",
    )

    clip_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    clip_bound = helper.make_tensor_value_info("runtime_min", TensorProto.FLOAT, [])
    clip_output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
    clip_max = _tensor("clip_max", np.array(1.0, dtype=np.float32))
    _save(
        helper.make_graph(
            [
                helper.make_node(
                    "Clip",
                    ["x", "runtime_min", "clip_max"],
                    ["y"],
                    name="runtime_clip_bound",
                )
            ],
            "runtime_clip_bound",
            [clip_x, clip_bound],
            [clip_output],
            initializer=[clip_max],
        ),
        output_dir / "clip_runtime_bound.onnx",
    )

    _save(
        helper.make_graph(
            [
                helper.make_node(
                    "Clip", ["x", "", "clip_max"], ["y"], name="missing_clip_bound"
                )
            ],
            "missing_clip_bound",
            [clip_x],
            [clip_output],
            initializer=[clip_max],
        ),
        output_dir / "clip_missing_bound.onnx",
    )

    _save(
        helper.make_graph(
            [
                helper.make_node(
                    "Clip",
                    ["x", "clip_min", "clip_max"],
                    ["y"],
                    name="nonscalar_clip_bound",
                )
            ],
            "nonscalar_clip_bound",
            [clip_x],
            [clip_output],
            initializer=[
                _tensor("clip_min", np.array([-1.0], dtype=np.float32)),
                clip_max,
            ],
        ),
        output_dir / "clip_nonscalar_bound.onnx",
    )

    _save(
        helper.make_graph(
            [
                helper.make_node(
                    "Clip",
                    ["x", "clip_min", "clip_max"],
                    ["y"],
                    name="nonfinite_clip_bound",
                )
            ],
            "nonfinite_clip_bound",
            [clip_x],
            [clip_output],
            initializer=[
                _tensor("clip_min", np.array(-np.inf, dtype=np.float32)),
                clip_max,
            ],
        ),
        output_dir / "clip_nonfinite_bound.onnx",
    )

    _save(
        helper.make_graph(
            [
                helper.make_node(
                    "Clip",
                    ["x", "clip_min", "clip_max"],
                    ["y"],
                    name="reversed_clip_bound",
                )
            ],
            "reversed_clip_bound",
            [clip_x],
            [clip_output],
            initializer=[
                _tensor("clip_min", np.array(2.0, dtype=np.float32)),
                clip_max,
            ],
        ),
        output_dir / "clip_reversed_bound.onnx",
    )

    softmax_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    softmax_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    _save(
        helper.make_graph(
            [helper.make_node("Softmax", ["x"], ["y"], name="unsupported_softmax")],
            "unsupported_operator",
            [softmax_x],
            [softmax_y],
        ),
        output_dir / "unsupported_operator.onnx",
    )

    _save(
        helper.make_graph(
            [helper.make_node("Relu", ["x"], ["y"], name="relu")],
            "wrong_opset",
            [softmax_x],
            [softmax_y],
        ),
        output_dir / "wrong_opset.onnx",
        opset=17,
    )

    auto_pad_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    auto_pad_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])
    _save(
        helper.make_graph(
            [
                helper.make_node(
                    "Conv", ["x", "w"], ["y"], name="auto_pad_conv", auto_pad="SAME_UPPER"
                )
            ],
            "unsupported_attribute",
            [auto_pad_x],
            [auto_pad_y],
            initializer=[_tensor("w", np.ones((1, 1, 3, 3), dtype=np.float32))],
        ),
        output_dir / "unsupported_attribute.onnx",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=pathlib.Path)
    generate(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
