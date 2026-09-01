#!/usr/bin/env python3
"""Import the statically shaped f32 BuddyGraph ONNX subset into BGraph MLIR."""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import onnx
from mlir import ir
from mlir.dialects import func
from onnx import TensorProto, numpy_helper


SUPPORTED_OPSET = 18
SUPPORTED_OPS = {
    "Conv",
    "BatchNormalization",
    "Clip",
    "Relu",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Reshape",
    "Transpose",
    "ReduceMean",
}


class ImportFailure(RuntimeError):
    """A deterministic, user-facing error for an unsupported ONNX construct."""


@dataclass(frozen=True)
class TensorInfo:
    shape: tuple[int, ...]


def _node_label(node: onnx.NodeProto, index: int) -> str:
    return node.name or f"{node.op_type}_{index}"


def _fail(message: str, node: onnx.NodeProto | None = None, index: int = 0) -> None:
    if node is None:
        raise ImportFailure(message)
    raise ImportFailure(f"node '{_node_label(node, index)}' ({node.op_type}): {message}")


def _attribute_map(node: onnx.NodeProto) -> dict[str, onnx.AttributeProto]:
    return {attribute.name: attribute for attribute in node.attribute}


def _ints_attribute(
    attributes: dict[str, onnx.AttributeProto], name: str, default: Sequence[int]
) -> list[int]:
    attribute = attributes.get(name)
    return list(default) if attribute is None else [int(value) for value in attribute.ints]


def _int_attribute(
    attributes: dict[str, onnx.AttributeProto], name: str, default: int
) -> int:
    attribute = attributes.get(name)
    return default if attribute is None else int(attribute.i)


def _float_attribute(
    attributes: dict[str, onnx.AttributeProto], name: str, default: float
) -> float:
    attribute = attributes.get(name)
    return default if attribute is None else float(attribute.f)


def _string_attribute(
    attributes: dict[str, onnx.AttributeProto], name: str, default: str
) -> str:
    attribute = attributes.get(name)
    return default if attribute is None else attribute.s.decode("utf-8")


class BGraphImporter:
    def __init__(self, model: onnx.ModelProto, function_name: str, scalar_return: bool):
        self.model = model
        self.graph = model.graph
        self.function_name = function_name
        self.scalar_return = scalar_return
        self.initializers = {initializer.name: initializer for initializer in self.graph.initializer}
        self.tensor_info = self._collect_tensor_info()
        self.context = ir.Context()
        self.context.allow_unregistered_dialects = True
        self.values: dict[str, ir.Value] = {}

    def _collect_tensor_info(self) -> dict[str, TensorInfo]:
        info: dict[str, TensorInfo] = {}
        declarations = list(self.graph.input) + list(self.graph.value_info) + list(self.graph.output)
        for value in declarations:
            tensor_type = value.type.tensor_type
            if tensor_type.elem_type != TensorProto.FLOAT:
                raise ImportFailure(
                    f"value '{value.name}' must be a tensor<float32>; "
                    f"found ONNX element type {tensor_type.elem_type}"
                )
            shape: list[int] = []
            for dimension in tensor_type.shape.dim:
                if not dimension.HasField("dim_value"):
                    raise ImportFailure(
                        f"value '{value.name}' has a dynamic or symbolic shape; "
                        "the BuddyGraph MVP requires static ranked tensors"
                    )
                shape.append(int(dimension.dim_value))
            info[value.name] = TensorInfo(tuple(shape))
        for name, initializer in self.initializers.items():
            if initializer.data_type != TensorProto.FLOAT and name not in self._attribute_initializers():
                raise ImportFailure(
                    f"initializer '{name}' must be float32 unless it supplies compile-time metadata"
                )
            if initializer.data_type == TensorProto.FLOAT:
                info[name] = TensorInfo(tuple(int(dim) for dim in initializer.dims))
        return info

    def _attribute_initializers(self) -> set[str]:
        names: set[str] = set()
        for node in self.graph.node:
            names.update(self._node_attribute_initializers(node))
        return names

    @staticmethod
    def _node_attribute_initializers(node: onnx.NodeProto) -> set[str]:
        if node.op_type == "Reshape" and len(node.input) > 1:
            return {node.input[1]}
        if node.op_type == "ReduceMean" and len(node.input) > 1 and node.input[1]:
            return {node.input[1]}
        if node.op_type == "Clip":
            return {name for name in node.input[1:3] if name}
        return set()

    def _tensor_type(self, name: str) -> ir.RankedTensorType:
        tensor = self.tensor_info.get(name)
        if tensor is None:
            raise ImportFailure(
                f"shape inference did not produce static type information for value '{name}'"
            )
        return ir.RankedTensorType.get(tensor.shape, ir.F32Type.get())

    def _value(self, name: str, node: onnx.NodeProto, index: int) -> ir.Value:
        value = self.values.get(name)
        if value is None:
            _fail(f"input '{name}' is unavailable; graph must be topologically sorted", node, index)
        return value

    def _constant_ints(self, name: str, node: onnx.NodeProto, index: int) -> list[int]:
        initializer = self.initializers.get(name)
        if initializer is None:
            _fail(f"'{name}' must be a constant initializer", node, index)
        array = numpy_helper.to_array(initializer)
        if array.dtype.kind not in {"i", "u"}:
            _fail(f"constant '{name}' must contain integers", node, index)
        return [int(value) for value in np.asarray(array).reshape(-1)]

    def _constant_scalar_f32(
        self, name: str, node: onnx.NodeProto, index: int
    ) -> float:
        initializer = self.initializers.get(name)
        if initializer is None:
            _fail(f"'{name}' must be a constant initializer", node, index)
        if initializer.data_type != TensorProto.FLOAT:
            _fail(f"constant '{name}' must be float32", node, index)
        array = np.asarray(numpy_helper.to_array(initializer))
        if array.ndim != 0:
            _fail(f"constant '{name}' must be a scalar", node, index)
        value = float(array)
        if not np.isfinite(value):
            _fail(f"constant '{name}' must be finite", node, index)
        return value

    @staticmethod
    def _i64_array(values: Iterable[int]) -> ir.DenseI64ArrayAttr:
        return ir.DenseI64ArrayAttr.get([int(value) for value in values])

    @staticmethod
    def _layout_nchw() -> ir.Attribute:
        return ir.Attribute.parse("#bgraph.layout<nchw>")

    def _create_bgraph(
        self,
        name: str,
        result_name: str,
        operands: Sequence[ir.Value],
        attributes: dict[str, ir.Attribute],
        location: ir.Location,
    ) -> ir.Value:
        operation = ir.Operation.create(
            f"bgraph.{name}",
            results=[self._tensor_type(result_name)],
            operands=operands,
            attributes=attributes,
            loc=location,
        )
        return operation.results[0]

    def _import_initializer(self, name: str) -> None:
        initializer = self.initializers[name]
        if initializer.data_type != TensorProto.FLOAT:
            return
        array = np.asarray(numpy_helper.to_array(initializer), dtype=np.float32)
        tensor_type = self._tensor_type(name)
        if array.ndim == 0:
            value_attr = ir.DenseElementsAttr.get_splat(
                tensor_type, ir.FloatAttr.get(ir.F32Type.get(), float(array))
            )
        else:
            array = np.ascontiguousarray(array)
            value_attr = ir.DenseElementsAttr.get(
                array, type=ir.F32Type.get(), shape=array.shape
            )
        operation = ir.Operation.create(
            "arith.constant",
            results=[tensor_type],
            attributes={"value": value_attr},
            loc=ir.Location.name(f"initializer:{name}"),
        )
        self.values[name] = operation.results[0]

    def _validate_node_contract(self, node: onnx.NodeProto, index: int) -> None:
        if node.domain not in {"", "ai.onnx"}:
            _fail(f"unsupported domain '{node.domain}'", node, index)
        if node.op_type not in SUPPORTED_OPS:
            _fail(f"unsupported operator; supported operators are {sorted(SUPPORTED_OPS)}", node, index)
        if len(node.output) != 1:
            _fail("exactly one output is required by the BuddyGraph MVP", node, index)

    def _import_node(self, node: onnx.NodeProto, index: int) -> None:
        self._validate_node_contract(node, index)
        attributes = _attribute_map(node)
        operands = [
            self._value(name, node, index)
            for name in node.input
            if name and name not in self._node_attribute_initializers(node)
        ]
        output = node.output[0]
        location = ir.Location.name(_node_label(node, index))

        if node.op_type == "Conv":
            if len(node.input) not in {2, 3}:
                _fail("expected input, filter, and optional bias", node, index)
            auto_pad = _string_attribute(attributes, "auto_pad", "NOTSET")
            if auto_pad != "NOTSET":
                _fail(f"auto_pad='{auto_pad}' is unsupported; use explicit pads", node, index)
            group = _int_attribute(attributes, "group", 1)
            if group != 1:
                _fail("grouped convolution is outside the MVP (group must be 1)", node, index)
            kernel_shape = _ints_attribute(attributes, "kernel_shape", [])
            if kernel_shape and len(kernel_shape) != 2:
                _fail("only two-dimensional convolution is supported", node, index)
            value = self._create_bgraph(
                "conv2d",
                output,
                operands,
                {
                    "strides": self._i64_array(_ints_attribute(attributes, "strides", [1, 1])),
                    "pads": self._i64_array(_ints_attribute(attributes, "pads", [0, 0, 0, 0])),
                    "dilations": self._i64_array(_ints_attribute(attributes, "dilations", [1, 1])),
                    "groups": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), group),
                    "layout": self._layout_nchw(),
                },
                location,
            )
        elif node.op_type == "BatchNormalization":
            if len(node.input) != 5:
                _fail("inference mode requires input, scale, bias, mean, and variance", node, index)
            if _int_attribute(attributes, "training_mode", 0) != 0:
                _fail("training_mode is unsupported", node, index)
            value = self._create_bgraph(
                "batch_norm",
                output,
                operands,
                {
                    "epsilon": ir.FloatAttr.get(
                        ir.F64Type.get(), _float_attribute(attributes, "epsilon", 1.0e-5)
                    ),
                    "layout": self._layout_nchw(),
                },
                location,
            )
        elif node.op_type in {"Relu", "Add", "Sub", "Mul", "Div"}:
            expected = 1 if node.op_type == "Relu" else 2
            if len(operands) != expected:
                _fail(f"expected {expected} tensor operand(s)", node, index)
            value = self._create_bgraph(node.op_type.lower(), output, operands, {}, location)
        elif node.op_type == "Clip":
            if len(node.input) != 3 or not all(node.input):
                _fail(
                    "expected data, constant scalar min, and constant scalar max inputs",
                    node,
                    index,
                )
            minimum = self._constant_scalar_f32(node.input[1], node, index)
            maximum = self._constant_scalar_f32(node.input[2], node, index)
            if minimum > maximum:
                _fail("min bound must be less than or equal to max bound", node, index)
            value = self._create_bgraph(
                "clamp",
                output,
                operands,
                {
                    "min_value": ir.FloatAttr.get(ir.F32Type.get(), minimum),
                    "max_value": ir.FloatAttr.get(ir.F32Type.get(), maximum),
                },
                location,
            )
        elif node.op_type == "Reshape":
            if len(node.input) != 2:
                _fail("expected data and a constant shape input", node, index)
            if _int_attribute(attributes, "allowzero", 0) != 0:
                _fail("allowzero=1 is unsupported", node, index)
            shape = self._constant_ints(node.input[1], node, index)
            inferred_shape = self.tensor_info[output].shape
            # Store the resolved static shape, not ONNX's 0/-1 shorthand.
            if any(dimension <= 0 for dimension in shape):
                shape = list(inferred_shape)
            value = self._create_bgraph(
                "reshape", output, operands, {"shape": self._i64_array(shape)}, location
            )
        elif node.op_type == "Transpose":
            rank = len(self.tensor_info[node.input[0]].shape)
            permutation = _ints_attribute(attributes, "perm", list(reversed(range(rank))))
            value = self._create_bgraph(
                "transpose",
                output,
                operands,
                {"permutation": self._i64_array(permutation)},
                location,
            )
        elif node.op_type == "ReduceMean":
            if len(node.input) > 2:
                _fail("expected data and optional constant axes", node, index)
            rank = len(self.tensor_info[node.input[0]].shape)
            axes = (
                self._constant_ints(node.input[1], node, index)
                if len(node.input) == 2 and node.input[1]
                else list(range(rank))
            )
            if not axes:
                if _int_attribute(attributes, "noop_with_empty_axes", 0) != 0:
                    _fail("noop_with_empty_axes=1 is unsupported", node, index)
                axes = list(range(rank))
            value = self._create_bgraph(
                "reduce_mean",
                output,
                operands,
                {
                    "axes": self._i64_array(axes),
                    "keep_dims": ir.BoolAttr.get(bool(_int_attribute(attributes, "keepdims", 1))),
                },
                location,
            )
        else:  # Kept defensive if SUPPORTED_OPS and dispatch drift apart.
            _fail("operator has no importer implementation", node, index)

        self.values[output] = value

    def import_module(self) -> ir.Module:
        with self.context, ir.Location.unknown():
            module = ir.Module.create()
            graph_inputs = [value for value in self.graph.input if value.name not in self.initializers]
            input_types = [self._tensor_type(value.name) for value in graph_inputs]
            output_types: list[ir.Type] = [self._tensor_type(value.name) for value in self.graph.output]
            if self.scalar_return:
                if len(output_types) != 1 or output_types[0].rank != 0:
                    raise ImportFailure("--scalar-return requires exactly one rank-0 tensor graph output")
                output_types = [ir.F32Type.get()]

            with ir.InsertionPoint(module.body):
                function = func.FuncOp(self.function_name, (input_types, output_types), visibility="public")
            entry = function.add_entry_block()
            with ir.InsertionPoint(entry):
                for graph_input, argument in zip(graph_inputs, entry.arguments):
                    self.values[graph_input.name] = argument
                for name in self.initializers:
                    self._import_initializer(name)
                for index, node in enumerate(self.graph.node):
                    self._import_node(node, index)

                returns = [self.values[value.name] for value in self.graph.output]
                if self.scalar_return:
                    extract = ir.Operation.create(
                        "tensor.extract",
                        results=[ir.F32Type.get()],
                        operands=returns,
                        loc=ir.Location.name("scalar_return"),
                    )
                    returns = [extract.results[0]]
                func.ReturnOp(returns)

            if not module.operation.verify():
                raise ImportFailure("the generated MLIR module failed verification")
            return module


def _validate_model(model: onnx.ModelProto) -> onnx.ModelProto:
    default_opsets = [opset.version for opset in model.opset_import if opset.domain in {"", "ai.onnx"}]
    if default_opsets != [SUPPORTED_OPSET]:
        raise ImportFailure(
            f"expected exactly ONNX opset {SUPPORTED_OPSET} for the default domain; "
            f"found {default_opsets or 'none'}"
        )
    unsupported_domains = [opset.domain for opset in model.opset_import if opset.domain not in {"", "ai.onnx"}]
    if unsupported_domains:
        raise ImportFailure(f"unsupported opset domains: {unsupported_domains}")
    onnx.checker.check_model(model)
    return onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path, help="ONNX model to import")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="output MLIR file (default: stdout)")
    parser.add_argument("--function-name", default="main_graph", help="generated func.func symbol")
    parser.add_argument(
        "--scalar-return",
        action="store_true",
        help="extract a sole rank-0 tensor result to f32 for mlir-runner",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        model = _validate_model(onnx.load(arguments.model, load_external_data=True))
        module = BGraphImporter(model, arguments.function_name, arguments.scalar_return).import_module()
        # Preserve ONNX node names as MLIR locations in the serialized module.
        output = module.operation.get_asm(enable_debug_info=True) + "\n"
        if arguments.output:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
    except (
        ImportFailure,
        onnx.checker.ValidationError,
        onnx.shape_inference.InferenceError,
        OSError,
        ValueError,
        ir.MLIRError,
    ) as error:
        print(f"buddygraph-import-onnx: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
